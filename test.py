import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline # EulerDiscreteScheduler

from PIL import Image
import numpy as np

from src.stable_diffusion import StableDiffusionModel
from max.engine import InferenceSession
from src.model_config import StableDiffusionConfig

# --- SDXLConfig and Diffusers config only; MAX PipelineConfig is NOT used for SDXL ---

device = "cuda"

# Load the full pipeline from HuggingFace
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", dtype=torch.float32)
pipe = pipe.to(device)

image = pipe(
    prompt="A beautiful landscape painting",
    negative_prompt="blurry",
    height=512,
    width=512,
    num_inference_steps=50
).images[0]
image.save("output_landscape_huggingface_pipeline.png")

# Extract UNet and VAE
unet = pipe.unet
vae = pipe.vae

print("scheduler:", pipe.scheduler)

# Instantiate your SDXLModel with all other components
model = StableDiffusionModel(
    text_encoder=pipe.text_encoder,
    # text_encoder_2=pipe.text_encoder, # initially was trying to do SDXL
    tokenizer=pipe.tokenizer,
    # tokenizer_2=pipe.tokenizer, # initially was trying to do SDXL
    scheduler=pipe.scheduler,
    image_encoder=getattr(pipe, 'image_encoder', None),
    feature_extractor=getattr(pipe, 'feature_extractor', None),
    force_zeros_for_empty_prompt=True,
    device=device
)
# Set unet and vae as attributes
model.unet = unet
model.vae = vae

# Example inference
if __name__ == "__main__":
    result = model.execute(
        prompt="A beautiful landscape painting",
        negative_prompt="blurry",
        height=512,
        width=512,
        num_inference_steps=50
    )
    print("Output tensor shape:", result.shape)  
      
    # Convert MAX tensor to numpy, then to PIL Image
    result_np = result.to_numpy()
    print("Result min/max before normalization:", np.nanmin(result_np), np.nanmax(result_np))
    if np.isnan(result_np).any():
        print("NaNs detected in result before normalization!")
    # Normalize to 0-255 range and convert to uint8
    result_np = ((result_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    print("Result min/max after normalization:", np.nanmin(result_np), np.nanmax(result_np))
    if np.isnan(result_np).any():
        print("NaNs detected in result after normalization!")
    
    # Convert to PIL Image (assuming shape is [1, 3, H, W] or [3, H, W])
    if result_np.shape[0] == 1:
        result_np = result_np[0]  # Remove batch dimension
    result_np = np.transpose(result_np, (1, 2, 0))  # CHW to HWC
    
    image = Image.fromarray(result_np)
    image.save("output_landscape.png")
    print("Image saved as output_landscape.png")
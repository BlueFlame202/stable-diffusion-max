import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline # EulerDiscreteScheduler

from PIL import Image
import numpy as np

from src.stable_diffusion import StableDiffusionModel
from src.clip import CLIPTextEncoder
from src.model_config import StableDiffusionConfig

from max.engine import InferenceSession
from max.driver import CPU, Accelerator
from max.dtype import DType

from max.pipelines.lib import (
    TextTokenizer,
)

import asyncio

# --- SDXLConfig and Diffusers config only; MAX PipelineConfig is NOT used for SDXL ---

device = "cuda"

# Load the full pipeline from HuggingFace
# CompVis/stable-diffusion-v1-4
# lambdalabs/sd-pokemon-diffusers
# common-canvas is technically a different model family, but seems to give interesting and useful results!
# common-canvas/CommonCanvas-S-C
# common-canvas/CommonCanvas-S-NC
model_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_path, dtype=torch.float32)
pipe = pipe.to(device)

prompt = "A scenic view of a lake"
negative_prompt = "blurry"

# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     height=512,
#     width=512,
#     num_inference_steps=50
# ).images[0]
# image.save("output_huggingface_pipeline.png")

# Extract UNet and VAE
unet = pipe.unet
vae = pipe.vae

text_encoder = CLIPTextEncoder(
    devices=[Accelerator()],
    hidden_dim=pipe.text_encoder.config.hidden_size,
    n_layers=pipe.text_encoder.config.num_hidden_layers,
    n_heads=pipe.text_encoder.config.num_attention_heads,
    vocab_size=pipe.text_encoder.config.vocab_size,
    max_len=pipe.text_encoder.config.max_position_embeddings,
    dtype=DType.float32
)
text_encoder.load_clip_state(pipe.text_encoder.state_dict())
print("Loaded state dict from the pipe model!") # TODO: load from safetensors file instead of from the pipe model

# Instantiate your SDXLModel with all other components
model = StableDiffusionModel(
    text_encoder=text_encoder,
    # text_encoder_2=pipe.text_encoder, # initially was trying to do SDXL
    tokenizer=TextTokenizer("/home/ubuntu/CommonCanvas-S-C/tokenizer", trust_remote_code=True),
    # tokenizer_2=pipe.tokenizer, # initially was trying to do SDXL
    scheduler=pipe.scheduler,
    device=device
)
# Set unet and vae as attributes
model.unet = unet
model.vae = vae
print(unet)

# Example inference
if __name__ == "__main__":
    result = asyncio.run(model.execute(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=50
    ))
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
    image.save("output.png")
    print("Image saved as output.png")
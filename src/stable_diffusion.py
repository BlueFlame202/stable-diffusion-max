import numpy as np
import torch
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from max.driver import Tensor

class CLIPTokenizer:
    def __init__():
        # TODO
        pass

    def from_weights():
        # TODO
        pass

    def __call__():
        # TODO
        pass


class StableDiffusionModel:
    def __init__(
        self,
        text_encoder: CLIPTextModel = None,
        # text_encoder_2: CLIPTextModelWithProjection = None,
        tokenizer: CLIPTokenizer = None,
        # tokenizer_2: CLIPTokenizer = None,
        scheduler=None,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        device: str = "cuda",
    ):
        self.text_encoder = text_encoder
        # self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        # self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.force_zeros_for_empty_prompt = force_zeros_for_empty_prompt
        self.device = device
        # Move models to the correct device
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self.device)
        # if self.text_encoder_2 is not None:
        #     self.text_encoder_2 = self.text_encoder_2.to(self.device)
        # unet/vae will be loaded from model_config.py

    def encode_prompt(self, prompt, negative_prompt, device):
        # Tokenize and encode with first encoder
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        negative_prompt_tokens = self.tokenizer(negative_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        with torch.no_grad():
            prompt_embeds = self.text_encoder(prompt_tokens["input_ids"].to(device))[0]
            negative_prompt_embeds = self.text_encoder(negative_prompt_tokens["input_ids"].to(device))[0]
        # Return as numpy for MAX
        return prompt_embeds.cpu().numpy(), negative_prompt_embeds.cpu().numpy()

    def prepare_latents(self, batch_size, channels, height, width, dtype):
        latents = np.random.randn(batch_size, channels, height, width).astype(dtype)
        latents = latents * self.scheduler.init_noise_sigma # supposedly crucial for SD v1.x
        return Tensor.from_numpy(latents)

    def execute(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_inference_steps: int = 50,
        **kwargs
    ):
        batch_size = 1
        # 1. Encode prompt
        prompt_embeds_np, negative_prompt_embeds_np = self.encode_prompt(prompt, negative_prompt, self.device)
        prompt_embeds_max = Tensor.from_numpy(prompt_embeds_np)
        negative_prompt_embeds_max = Tensor.from_numpy(negative_prompt_embeds_np)
        # 2. Prepare latents
        latents_max = self.prepare_latents(batch_size, 4, height // 8, width // 8, np.float32)
        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # 4. Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # MAX → PyTorch # TODO: use the dlpack thing to keep stuff on the same device
            latents_torch = torch.from_dlpack(latents_max).to(self.device)
            latents_torch = self.scheduler.scale_model_input(latents_torch, t)
            print(f"Step {t}: latents min/max/std: {latents_torch.min().item()}/{latents_torch.max().item()}/{latents_torch.std().item()}")
            if torch.isnan(latents_torch).any():
                print("NaNs detected in latents before scaling!")
            prompt_embeds_torch = torch.from_dlpack(prompt_embeds_max).to(self.device)
            # Run UNet
            with torch.no_grad():
                noise_pred = self.unet(
                    latents_torch,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds_torch,
                ).sample
            # PyTorch → MAX
            # noise_pred_max = Tensor.from_dlpack(noise_pred)
            # Scheduler step (PyTorch or MAX)
            latents_torch = self.scheduler.step(noise_pred, t, latents_torch).prev_sample # [0] # <- should be [0] when return_dict is False, but default is True https://huggingface.co/docs/diffusers/v0.36.0/en/api/schedulers/pndm#diffusers.schedulers.scheduling_utils.SchedulerOutput
            latents_max = Tensor.from_dlpack(latents_torch)
        # 5. VAE decode
        latents_torch = torch.from_dlpack(latents_max).to(self.device)
        latents_torch = latents_torch / 0.18215 # check why its division by 0.18215 that works here and not / self.scheduler.init_noise_sigma
        with torch.no_grad():
            image = self.vae.decode(latents_torch).sample
        print("Image min/max:", image.min().item(), image.max().item())
        if torch.isnan(image).any():
            print("NaNs detected in VAE output!")
        image_max = Tensor.from_numpy(image.cpu().numpy())
        return image_max
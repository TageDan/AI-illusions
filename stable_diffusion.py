import numpy as np

import os

import torch

from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from torchvision import transforms as tfms
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging


class StableDiffusion:
    def __init__(self, steps = 30):
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        diffusion_model = "johnrobinsn/miniSD"
        text_model = "openai/clip-vit-large-patch14"
        
        self.height = 256
        self.width = 256

        self.device = torch_device
        self.vae = AutoencoderKL.from_pretrained(diffusion_model, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(text_model)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model)

        self.unet = UNet2DConditionModel.from_pretrained(
            diffusion_model, subfolder="unet"
        )

        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        self.guidance = 7.5
        self.step = 0
        self.steps = steps

    def set_prompt(self, prompt_str: str):
        prompt = [prompt_str]
        num_inference_steps = self.steps
        batch_size= 1

        text_input = self.tokenizer(
            prompt,
            padding = "max_length",
            max_length = self.tokenizer.model_max_length,
            truncation = True,
            return_tensors = "pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]


        max_length = text_input.input_ids.shape[-1]

        uncond_input = self.tokenizer(
            [""]*batch_size,
            padding = "max_length",
            max_length = max_length,
            return_tensors = "pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        def set_timesteps(scheduler, num_inf_steps):
            scheduler.set_timesteps(num_inf_steps)
            scheduler.timesteps = scheduler.timesteps.to(self.device)

        set_timesteps(self.scheduler, num_inference_steps)

    def pil_to_latent(self, input_image):
        with torch.no_grad():
            latent = self.vae.encode(tfms.ToTensor()(input_image).unsqueeze(0).to(self.device)*2-1)

        return 0.18215 * latent.latent_dist.sample()

    def latents_to_pil(self, latents):

        image = 1/0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(image).sample

        image = (image / 2 +0.5).clamp(0,1)

        image = image.detach().cpu().permute(0,2,3,1).numpy()

        images = (image*255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


    def add_noise(self, latent):
        noise = torch.randn_like(latent)

        sampling_step = self.step

        encoded_and_noised_img = self.scheduler.add_noise(
            latent,
            noise,
            timesteps = torch.tensor([self.scheduler.timesteps[self.step]])
        )

        return encoded_and_noised_img

    def train_step(self, latents):
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            if i == self.step:
                latent_model_input = torch.cat([latents]*2)
                sigma = self.scheduler.sigmas[i]
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                  
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states = self.text_embeddings
                    )["sample"]
                 
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.step += 1
        return latents

    def prepare_latent(self, latent):
        latent = latent.to(self.device)
        latent = latent*self.scheduler.init_noise_sigma
        return latent

        
            
            

        
if __name__ == "__main__":
    # print("test init")
    # sd = StableDiffusion()
    # print("init passed")
    # print("------------")
    # print("add noise test")
    # dog = Image.open("dog.jpg")
    # sd.latents_to_pil(sd.add_noise(sd.pil_to_latent(dog)))[0].show("noisy dog")
    # dog.show("dog")
    # print("add noise test passed ??")
    # print("-----------")
    print("Image generation test")
    steps = 10
    prompt = "A watercolor painting of a stray black and white dog"
    sd = StableDiffusion(steps=steps)
    sd.set_prompt(prompt)
    image_latent = torch.randn(
        (1, sd.unet.in_channels, 256//8, 256//8),
    )
    image_latent = sd.prepare_latent(image_latent)
    for i in range(0, steps):
      # image_latent = sd.add_noise(image_latent)
      image_latent = sd.train_step(image_latent)
      image = sd.latents_to_pil(image_latent)[0]
      if not i % 5:
          sd.latents_to_pil(image_latent)[0].show("progress")
      

    # image_latent = sd.add_noise(image_latent)
    # image_latent = sd.train_step(image_latent)
        
    sd.latents_to_pil(image_latent)[0].show("hopefully moon")


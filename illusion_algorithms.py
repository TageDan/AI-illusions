from image_networks import SimpleTrainableImage
from PIL import Image, ImageFilter
from diffusers import StableDiffusionImg2ImgPipeline
from stable_diffusion import StableDiffusion
import torch


def flippy_image_from_image(path, epoch):
    target = Image.open(path).resize((200, 200))
    flipped_target = target.rotate(180)

    res = SimpleTrainableImage((200, 200), 100, 128, 10.0)
    flipped_target_im = flipped_target.load()
    target_im = target.load()

    def normal(x, y):
        p = target_im[x, y]
        return (p[0] / 255, p[1] / 255, p[2] / 255)

    def flipped(x, y):
        p = flipped_target_im[x, y]
        return (p[0] / 255, p[1] / 255, p[2] / 255)

    for i in range(0, epoch * 2):
        if i % 2 == 0:
            res.train(normal)
        else:
            res.train(flipped)

    return res


def flippy_image_from_prompts(prompt1, prompt2, epochs):
    # create stable diffusion model for each arrangement
    sd1 = StableDiffusion(steps=epochs)
    sd2 = StableDiffusion(steps=epochs)
    
    # load prompts for arrangement
    sd1.set_prompt(prompt1)
    sd2.set_prompt(prompt2)

    res = SimpleTrainableImage((256,256), 100, 128, 10.0)

    im = None
    flipped_im = None 

    def normal(x,y):
        p = im.load()[x,y]
        return (p[0] / 255, p[1] / 255, p[2] / 255)

    def flipped(x,y):
        p = flipped_im.load()[x,y]
        return (p[0] / 255, p[1] / 255, p[2] / 255)

    latent1 = torch.randn(
        (1, sd.unet.in_channels, 256//8, 256//8),
    )
    latent2 = torch.randn(
        (1, sd.unet.in_channels, 256//8, 256//8),
    )
    for i in range(0, epochs):
        latent1 = sd1.train_step(latent1)
        im = sd1.latents_to_pil(latent1)[0]
        latent2 = sd2.train_step(latent2)
        flipped_im = sd2.latents_to_pil(latent2)[0].rotate(180)
        for _ in range(0,10):
            res.train(normal)
            res.train(flipped)
        image = res.get_image()
        latent1 = sd1.pil_to_latent(image)
        latent2 = sd2.pil_to_latent(image.rotate(180))
        if not i%2:
            im.show()
            flipped_im.show()
            image.show("hi")

    
    
if __name__ == "__main__":
    # flippy_image_from_image("dog.jpg", 50).get_image().show()
    flippy_image_from_prompts(
        "A watercolor painting of a stray black and white dog", "A watercolor painting of a stray black and white dog", 6
    ).get_image().show("flippy_ai_test")

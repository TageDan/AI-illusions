from image_networks import SimpleTrainableImage
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
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

    res = SimpleTrainableImage((160, 160), 100, 128, 10.0)

    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, requires_safety_checker=False, safety_checker=None
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    target = Image.new("RGB", (160, 160))
    flipped_target = Image.new("RGB", (160, 160))

    def normal(x, y):
        c = target.load()[x, y]
        return (c[0] / 255, c[1] / 255, c[2] / 255)

    def flipped(x, y):
        c = flipped_target.load()[x, y]
        return (c[0] / 255, c[1] / 255, c[2] / 255)

    def denoise_current(pipe, steps=100):
        im = res.get_image()
        flipped_im = im.rotate(180)
        strength = 0.85
        guidance_scale = 9.0
        target = pipe(
            prompt=prompt1,
            image=im,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        ).images[0]
        flipped_target = (
            pipe(
                prompt=prompt2,
                image=flipped_im,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
            )
            .images[0]
            .rotate(180)
        )
        return target, flipped_target

    for i in range(0, epochs[0]):
        target, flipped_target = denoise_current(pipe)
        for _ in range(0, epochs[1]):
            res.train(normal)
            res.train(flipped)
        res.get_image().show()
        target.show()
        flipped_target.show()

    return res


if __name__ == "__main__":
    # flippy_image_from_image("dog.jpg", 50).get_image().show()
    flippy_image_from_prompts(
        "a happy golden retreiver", "a happy golden retreiver", (30, 15)
    ).get_image().show("flippy_ai_test")

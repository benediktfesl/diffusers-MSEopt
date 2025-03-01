from diffusers import UNet2DModel
from PIL import Image
import torch
from scheduling_dmse import DMSEScheduler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
    scheduler.set_timesteps(5)

    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size), device=device)
    input = noise

    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model(input, t).sample
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            input = prev_noisy_sample

    image = (input / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))
    image

if __name__ == '__main__':
    main()
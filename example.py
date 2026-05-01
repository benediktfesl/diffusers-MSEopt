"""
Example: MSE-optimal denoising with DMSEScheduler.

Demonstrates two usage modes:
  1. Denoising a noisy observation at a known SNR (primary use case).
  2. Unconditional generation (full reverse chain, deterministic).
"""

from diffusers import UNet2DModel
from PIL import Image
import torch
from diffusers_dmse import DMSEScheduler


def denoising_example(snr_dB: float = 10.0, output_path: str = "output_denoised.png"):
    """
    Denoise a noisy observation using SNR-matched reverse diffusion (Eq. 12 of the paper).

    Args:
        snr_dB: SNR of the noisy input in dB.
        output_path: Where to save the output image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)

    # Use full timestep grid for accurate SNR matching.
    # Fewer steps (e.g. 200) reduce runtime at the cost of SNR-matching precision.
    scheduler.set_timesteps(1000)

    sample_size = model.config.sample_size

    # Simulate a noisy observation: clean image + AWGN at the given SNR
    # Here we just use pure noise as a placeholder input.
    x = torch.randn((1, 3, sample_size, sample_size), device=device)

    # Find the starting timestep matching the observed SNR
    t_init, idx = scheduler.init_step(snr=snr_dB, is_logarithmic=True)
    print(
        f"SNR = {snr_dB} dB  →  starting timestep t = {t_init}  (index {idx}/{len(scheduler.timesteps)})"
    )

    for t in scheduler.timesteps[idx:]:
        with torch.no_grad():
            eps = model(x, t).sample
        x = scheduler.step(eps, t, x).prev_sample

    image = (x / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))
    image.save(output_path)
    print(f"Saved: {output_path}")


def generation_example(num_steps: int = 50, output_path: str = "output_generated.png"):
    """
    Unconditional image generation using the deterministic (no-resampling) reverse process.

    This runs the full reverse chain from t=T, equivalent to DDIM with eta=0
    but using the DDPM posterior mean formula.

    Args:
        num_steps: Number of reverse diffusion steps.
        output_path: Where to save the output image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
    scheduler.set_timesteps(num_steps)

    sample_size = model.config.sample_size
    x = torch.randn((1, 3, sample_size, sample_size), device=device)

    for t in scheduler.timesteps:
        with torch.no_grad():
            eps = model(x, t).sample
        x = scheduler.step(eps, t, x).prev_sample

    image = (x / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    denoising_example(snr_dB=10.0)
    generation_example(num_steps=50)

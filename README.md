# diffusers-dmse

MSE-optimal diffusion model scheduler (**DMSE**) for the [HuggingFace diffusers](https://github.com/huggingface/diffusers) library.

Inherits `DDPMScheduler` and modifies the reverse process to omit stochastic resampling,
yielding a deterministic path that converges to the conditional mean estimator (CME) —
the MSE-optimal denoiser.

## Paper

B. Fesl, B. Böck, F. Strasser, M. Baur, M. Joham, W. Utschick,
"On the Asymptotic Mean Square Error Optimality of Diffusion Models,"
*AISTATS 2025*.

[[arXiv](https://arxiv.org/abs/2403.02957)] [[OpenReview](https://openreview.net/forum?id=XrXlAYFpvR)] [[PMLR](https://proceedings.mlr.press/v258/fesl25a.html)]

## Installation

```bash
pip install diffusers-dmse
```

## Usage

### Denoising a noisy observation (primary use case)

Use `init_step()` to find the timestep matching the observed SNR, then run
the reverse process from that point. This implements Eq. (12) of the paper.

```python
from diffusers import UNet2DModel
from diffusers_dmse import DMSEScheduler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)

# set_timesteps must be called before init_step
scheduler.set_timesteps(1000)

# Find starting timestep matching the observed SNR (in dB)
t_init, idx = scheduler.init_step(snr=10.0, is_logarithmic=True)

x = noisy_observation  # your input tensor, shape (B, C, H, W)
for t in scheduler.timesteps[idx:]:
    with torch.no_grad():
        eps = model(x, t).sample
    x = scheduler.step(eps, t, x).prev_sample
```

### Unconditional generation (deterministic DDPM)

Drop-in replacement for `DDPMScheduler`. Runs the full reverse chain without noise,
equivalent to DDIM with `eta=0` using the DDPM posterior mean.

```python
from diffusers import UNet2DModel
from diffusers_dmse import DMSEScheduler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
scheduler.set_timesteps(50)

x = torch.randn((1, 3, 256, 256), device=device)
for t in scheduler.timesteps:
    with torch.no_grad():
        eps = model(x, t).sample
    x = scheduler.step(eps, t, x).prev_sample
```

## Key difference from DDPMScheduler

| | DDPM | DMSE |
|---|---|---|
| Reverse step | `x_{t-1} = µ_t(x_t) + σ_t·z`, `z~N(0,I)` | `x_{t-1} = µ_t(x_t)` |
| Stochastic | Yes | No |
| Optimal for | Generation diversity | MSE / denoising |
| Starting point | `t=T` (pure noise) | SNR-matched `t` via `init_step()` |

## Related repositories

- **[Diffusion_MSE](https://github.com/benediktfesl/Diffusion_MSE)**: Full source code for the AISTATS 2025 paper, including GMM, MNIST, and audio experiments.
- **[Diffusion_channel_est](https://github.com/benediktfesl/Diffusion_channel_est)**: Application of DMSE to MIMO channel estimation (IEEE Wireless Communications Letters, 2024). [[Paper](https://ieeexplore.ieee.org/document/10705115)]

## License

MIT License. See [LICENSE](LICENSE).

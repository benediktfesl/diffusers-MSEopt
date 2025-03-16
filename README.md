# diffusers-MSEopt (DMSE)

This repository integrates the diffusion model-based MSE-optimal denoising strategy called **DMSE** into the [diffusers pipeline](https://github.com/huggingface/diffusers). The MSE-optimal denoising strategy is based on the paper **On the Asymptotic Mean Square Error Optimality of Diffusion Models**  


## Paper Reference
B. Fesl, B. Böck, F. Strasser, M. Baur, M. Joham, W. Utschick, "On the Asymptotic Mean Square Error Optimality of Diffusion Models." In _Proceedings of the 28th International Conference on Artificial Intelligence and Statistics_, 2025.

[[arXiv](https://arxiv.org/abs/2403.02957)] [[OpenReview](https://openreview.net/forum?id=XrXlAYFpvR)]

## Abstract

Diffusion models (DMs) as generative priors have recently shown great potential for denoising tasks but lack theoretical understanding with respect to their mean square error (MSE) optimality. This paper proposes a novel denoising strategy inspired by the structure of the MSE-optimal conditional mean estimator (CME). The resulting DM-based denoiser can be conveniently employed using a pre-trained DM, being particularly fast by truncating reverse diffusion steps and not requiring stochastic re-sampling. We present a comprehensive (non-)asymptotic optimality analysis of the proposed diffusion-based denoiser, demonstrating polynomial-time convergence to the CME under mild conditions. Our analysis also derives a novel Lipschitz constant that depends solely on the DM's hyperparameters. Further, we offer a new perspective on DMs, showing that they inherently combine an asymptotically optimal denoiser with a powerful generator, modifiable by switching re-sampling in the reverse process on or off. The theoretical findings are thoroughly validated with experiments based on various benchmark datasets.

## Features
- Integrates a reverse process for MSE-optimal denoising into the [diffusers pipeline](https://github.com/huggingface/diffusers).
- Inherits DDPM scheduling with a modified reverse process, avoiding stochastic re-sampling.
- Fast denoising due to truncation of reverse diffusion steps.
- Theoretical validation based on the proposed MSE-optimal conditional mean estimator (CME).

## Related Repositories

1. **[Diffusion_MSE](https://github.com/benediktfesl/Diffusion_MSE)**: Source code of the original paper "On the Asymptotic Mean Square Error Optimality of Diffusion Models."
2. **[Diffusion_channel_est](https://github.com/benediktfesl/Diffusion_channel_est)**: Source code of the paper "Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation."
   - Application of the proposed MSE-optimal denosing strategy for wireless channel estimation. 
   - **Paper Reference**:  
     B. Fesl, M. Baur, F. Strasser, M. Joham, W. Utschick, "Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation," _IEEE Wireless Communications Letters_, vol. 13, no. 12, pp. 3493-3497, Dec. 2024, doi: 10.1109/LWC.2024.3474570.
      [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10705115)] [[arXiv](https://arxiv.org/pdf/2403.03545)]

## Requirements
This repository requires the `diffusers` package along with all its dependencies. You can install it via:

```bash
pip install --upgrade diffusers
```

## Example Code

Below is an example of using the MSE-optimal denoising strategy with the `diffusers` pipeline:

```python
from diffusers import UNet2DModel
from PIL import Image
import torch
from scheduling_dmse import DMSEScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
scheduler.set_timesteps(50)

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
```


## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more information.


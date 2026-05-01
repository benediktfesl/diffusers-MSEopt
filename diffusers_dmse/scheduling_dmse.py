# Copyright (c) 2024 Benedikt Fesl. MIT License.
# Paper: https://arxiv.org/abs/2403.02957

from diffusers import DDPMScheduler
import torch
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Optional, Tuple, Union


@dataclass
class DMSESchedulerOutput(BaseOutput):
    """
    Output of :class:`DMSEScheduler.step`.

    Args:
        prev_sample (:obj:`torch.Tensor` of shape ``(batch_size, num_channels, height, width)``):
            Denoised sample ``x_{t-1}``. Use as the next model input in the denoising loop.
        pred_original_sample (:obj:`torch.Tensor` of shape ``(batch_size, num_channels, height, width)``, *optional*):
            Predicted clean sample ``x_0`` based on the current model output.
            Useful for monitoring convergence or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DMSEScheduler(DDPMScheduler):
    """
    MSE-optimal diffusion model scheduler (DMSE).

    Inherits :class:`diffusers.DDPMScheduler` and modifies the reverse process to omit
    stochastic resampling. The resulting deterministic path converges to the conditional
    mean estimator (CME) — the MSE-optimal denoiser — as shown in:

        B. Fesl et al., "On the Asymptotic Mean Square Error Optimality of Diffusion Models,"
        AISTATS 2025. https://arxiv.org/abs/2403.02957

    Key difference from DDPMScheduler
    ----------------------------------
    Standard DDPM reverse step: ``x_{t-1} = µ_t(x_t) + σ_t * z``, where ``z ~ N(0,I)``.
    DMSE reverse step:          ``x_{t-1} = µ_t(x_t)``  (noise term omitted).

    This makes the reverse process deterministic and MSE-optimal for denoising.

    Two usage modes
    ---------------
    **Denoising a noisy observation** (primary use case, see paper):
        Use :meth:`init_step` to find the timestep whose noise level matches the
        observed SNR, then run the reverse process from that timestep only.
        ``set_timesteps`` must be called before ``init_step``.

        Example::

            scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
            model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
            scheduler.set_timesteps(1000)

            t_init, idx = scheduler.init_step(snr=10.0, is_logarithmic=True)

            x = noisy_observation  # your input tensor
            for t in scheduler.timesteps[idx:]:
                with torch.no_grad():
                    eps = model(x, t).sample
                x = scheduler.step(eps, t, x).prev_sample

    **Unconditional generation** (same interface as DDPMScheduler, but deterministic):
        Run the full reverse chain from ``t=T``. Equivalent to DDIM with ``eta=0``
        using the DDPM posterior mean formula.

        Example::

            scheduler = DMSEScheduler.from_pretrained("google/ddpm-cat-256")
            model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
            scheduler.set_timesteps(50)

            x = torch.randn((1, 3, 256, 256), device=device)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    eps = model(x, t).sample
                x = scheduler.step(eps, t, x).prev_sample
    """

    def __init__(self, **kwargs):
        # No @register_to_config here: DMSEScheduler introduces no new config
        # parameters, so DDPMScheduler's registration is sufficient.
        super().__init__(**kwargs)
        self.snrs = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        # snrs_dB ranges from -inf (t=T, pure noise) to +inf (t=0, clean signal)
        self.snrs_dB = 10.0 * torch.log10(self.snrs)
        self._dmse_timesteps_initialized = False

    def set_timesteps(self, *args, **kwargs):
        result = super().set_timesteps(*args, **kwargs)
        self._dmse_timesteps_initialized = True
        return result

    def init_step(self, snr: float, is_logarithmic: bool = True) -> Tuple[int, int]:
        """
        Find the starting timestep whose noise level matches the observed SNR.

        Implements Eq. (12) of https://arxiv.org/abs/2403.02957.
        The reverse process is then run from this timestep to ``t=0``.

        .. note::
            :meth:`set_timesteps` must be called before this method.

        Args:
            snr (float):
                Signal-to-noise ratio of the noisy observation.
            is_logarithmic (bool, *optional*, defaults to ``True``):
                If ``True``, ``snr`` is interpreted in dB (logarithmic scale).
                If ``False``, ``snr`` is interpreted in linear scale.

        Returns:
            t (int):
                The matched timestep value.
            idx (int):
                Index into ``scheduler.timesteps``; iterate over
                ``scheduler.timesteps[idx:]`` in the denoising loop.

        Raises:
            AttributeError: If :meth:`set_timesteps` has not been called yet.

        Example::

            scheduler.set_timesteps(1000)
            t_init, idx = scheduler.init_step(snr=10.0, is_logarithmic=True)
            for t in scheduler.timesteps[idx:]:
                ...
        """
        if not self._dmse_timesteps_initialized:
            raise AttributeError(
                "set_timesteps() must be called before init_step(). "
                "Example: scheduler.set_timesteps(1000)"
            )
        if is_logarithmic:
            t = int(torch.abs(self.snrs_dB - snr).argmin())
        else:
            t = int(torch.abs(self.snrs - snr).argmin())
        idx = int(torch.abs(self.timesteps - t).argmin())
        t = self.timesteps[idx]
        return t, idx

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DMSESchedulerOutput, Tuple]:
        """
        One reverse-diffusion step: compute ``x_{t-1}`` from ``x_t`` without adding noise.

        Computes the DDPM posterior mean ``µ_t`` (Eq. 7 of Ho et al., 2020) and returns
        it directly, omitting the stochastic term. This implements the DMSE update rule.

        Args:
            model_output (:obj:`torch.Tensor`):
                Direct output of the diffusion UNet at timestep ``t``.
                Interpretation depends on ``prediction_type`` in the scheduler config
                (``"epsilon"``, ``"sample"``, or ``"v_prediction"``).
            timestep (int):
                Current discrete timestep ``t``.
            sample (:obj:`torch.Tensor`):
                Current noisy sample ``x_t``.
            generator (:obj:`torch.Generator`, *optional*):
                Unused. Kept for API compatibility with :class:`diffusers.DDPMScheduler`.
            return_dict (bool, *optional*, defaults to ``True``):
                If ``True``, return :class:`DMSESchedulerOutput`.
                If ``False``, return a ``(prev_sample, pred_original_sample)`` tuple.

        Returns:
            :class:`DMSESchedulerOutput` or tuple:
                ``prev_sample`` is the denoised ``x_{t-1}``.
                ``pred_original_sample`` is the predicted clean ``x_0``.
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        # Handle learned variance models
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, _ = torch.split(model_output, sample.shape[1], dim=1)

        # 1. Compute alphas and betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. Predict clean sample x_0 from model output
        # Reference: Eq. (15) of Ho et al. (DDPM), https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type '{self.config.prediction_type}' must be one of "
                "'epsilon', 'sample', or 'v_prediction'."
            )

        # 3. Apply clipping / dynamic thresholding to x_0 if configured
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute DDPM posterior mean µ_t (Eq. 7 of Ho et al.)
        #    DMSE: return µ_t directly, WITHOUT adding the noise term σ_t * z
        coeff_x0 = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        coeff_xt = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = coeff_x0 * pred_original_sample + coeff_xt * sample

        if not return_dict:
            return (pred_prev_sample, pred_original_sample)
        return DMSESchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )

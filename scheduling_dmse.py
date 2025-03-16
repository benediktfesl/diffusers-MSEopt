from diffusers import DDPMScheduler
import torch
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config


@dataclass
class DMSESchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DMSEScheduler(DDPMScheduler):
    """
    DMSE: A custom scheduler inheriting from DDPMScheduler, modifying the reverse process
    based on "On the Asymptotic Mean Square Error Optimality of Diffusion Models."
    Link to paper: https://arxiv.org/abs/2403.02957
    """

    @register_to_config
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.snrs = self.alphas_cumprod / (1. - self.alphas_cumprod)
        self.snrs_dB = 10 * torch.log10(self.snrs)

    def init_step(self, snr, is_logarithmic=True):
        """
        Calculates the initial timestep for denoising a given observation according to eq. (12) in
        https://arxiv.org/abs/2403.02957.

        Args:
            snr (`float`):
                The SNR of the observation.
            is_logarithmic (`bool`, *optional*, defaults to `True`):
                Whether the SNR is given in logarithmic scale (dB) or in linear scale.

        Returns:
            'torch.Tensor':
                The initial timestep of the diffusion model for running the denoising process.
            'int':
                The index of the self.timesteps array to which the initial timestep corresponds.
        """
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
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # Leave out adding noise in reverse process

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DMSESchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

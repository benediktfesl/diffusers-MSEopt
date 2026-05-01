"""
Tests for DMSEScheduler.

All tests run on CPU without network access (no from_pretrained calls).
DMSEScheduler is instantiated directly using DDPMScheduler defaults.
"""

import pytest
import torch
from diffusers import DDPMScheduler

from diffusers_dmse import DMSEScheduler, DMSESchedulerOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduler():
    """Default DMSEScheduler instance (1000 training steps, linear schedule)."""
    return DMSEScheduler()


@pytest.fixture
def scheduler_with_timesteps():
    """DMSEScheduler with set_timesteps already called."""
    s = DMSEScheduler()
    s.set_timesteps(100)
    return s


@pytest.fixture
def ddpm_scheduler():
    """Matching DDPMScheduler for comparison tests."""
    return DDPMScheduler()


@pytest.fixture
def dummy_batch():
    """Small (B=1, C=3, H=8, W=8) tensors for fast tests."""
    torch.manual_seed(0)
    B, C, H, W = 1, 3, 8, 8
    model_output = torch.randn(B, C, H, W)
    sample = torch.randn(B, C, H, W)
    return model_output, sample


# ---------------------------------------------------------------------------
# 1. Import and instantiation
# ---------------------------------------------------------------------------


def test_import():
    from diffusers_dmse import DMSEScheduler, DMSESchedulerOutput  # noqa: F401


def test_instantiation(scheduler):
    assert isinstance(scheduler, DMSEScheduler)
    assert isinstance(scheduler, DDPMScheduler)


def test_snr_attributes_exist(scheduler):
    """snrs and snrs_dB must be computed on init."""
    assert hasattr(scheduler, "snrs")
    assert hasattr(scheduler, "snrs_dB")
    assert scheduler.snrs.shape == (1000,)
    assert scheduler.snrs_dB.shape == (1000,)


def test_snrs_are_positive(scheduler):
    """SNR (linear) must be strictly positive."""
    assert (scheduler.snrs > 0).all()


def test_snrs_dB_monotone(scheduler):
    """SNR in dB should decrease monotonically with timestep t
    (more noise at higher t means lower SNR)."""
    assert (scheduler.snrs_dB[:-1] > scheduler.snrs_dB[1:]).all()


# ---------------------------------------------------------------------------
# 2. init_step
# ---------------------------------------------------------------------------


def test_init_step_raises_without_set_timesteps(scheduler):
    """init_step must raise AttributeError if set_timesteps was not called."""
    with pytest.raises(AttributeError, match="set_timesteps"):
        scheduler.init_step(snr=10.0)


def test_init_step_returns_valid_types(scheduler_with_timesteps):
    t, idx = scheduler_with_timesteps.init_step(snr=10.0, is_logarithmic=True)
    assert isinstance(int(t), int)
    assert isinstance(idx, int)


def test_init_step_idx_in_bounds(scheduler_with_timesteps):
    _, idx = scheduler_with_timesteps.init_step(snr=10.0, is_logarithmic=True)
    assert 0 <= idx < len(scheduler_with_timesteps.timesteps)


def test_init_step_t_in_timesteps(scheduler_with_timesteps):
    """Returned timestep t must be an element of scheduler.timesteps."""
    t, idx = scheduler_with_timesteps.init_step(snr=10.0, is_logarithmic=True)
    assert t == scheduler_with_timesteps.timesteps[idx]


def test_init_step_linear_and_dB_both_valid(scheduler_with_timesteps):
    """Both linear and dB modes must return valid, in-bounds results."""
    snr_dB = 10.0
    snr_linear = 10.0 ** (snr_dB / 10.0)  # 10 dB = 10x linear
    t_log, idx_log = scheduler_with_timesteps.init_step(snr=snr_dB, is_logarithmic=True)
    t_lin, idx_lin = scheduler_with_timesteps.init_step(
        snr=snr_linear, is_logarithmic=False
    )
    n = len(scheduler_with_timesteps.timesteps)
    assert 0 <= idx_log < n
    assert 0 <= idx_lin < n
    # For SNR=10 (a clean round value), nearest-neighbor in dB and linear
    # space should agree or be within one step of each other.
    assert abs(idx_log - idx_lin) <= 1


def test_init_step_high_snr_gives_low_timestep(scheduler_with_timesteps):
    """High SNR (clean signal) should map to a low timestep (little noise)."""
    _, idx_high = scheduler_with_timesteps.init_step(snr=30.0)
    _, idx_low = scheduler_with_timesteps.init_step(snr=-10.0)
    assert (
        idx_high > idx_low
    )  # higher SNR → later start in timesteps array → higher idx


# ---------------------------------------------------------------------------
# 3. step() — output correctness
# ---------------------------------------------------------------------------


def test_step_output_type_dict(scheduler_with_timesteps, dummy_batch):
    model_output, sample = dummy_batch
    t = scheduler_with_timesteps.timesteps[10]
    out = scheduler_with_timesteps.step(model_output, t, sample, return_dict=True)
    assert isinstance(out, DMSESchedulerOutput)


def test_step_output_type_tuple(scheduler_with_timesteps, dummy_batch):
    model_output, sample = dummy_batch
    t = scheduler_with_timesteps.timesteps[10]
    out = scheduler_with_timesteps.step(model_output, t, sample, return_dict=False)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_step_output_shape(scheduler_with_timesteps, dummy_batch):
    model_output, sample = dummy_batch
    t = scheduler_with_timesteps.timesteps[10]
    out = scheduler_with_timesteps.step(model_output, t, sample)
    assert out.prev_sample.shape == sample.shape
    assert out.pred_original_sample.shape == sample.shape


def test_step_is_deterministic(scheduler_with_timesteps, dummy_batch):
    """DMSE step must be fully deterministic — no stochastic resampling."""
    model_output, sample = dummy_batch
    t = scheduler_with_timesteps.timesteps[10]
    out1 = scheduler_with_timesteps.step(model_output, t, sample)
    out2 = scheduler_with_timesteps.step(model_output, t, sample)
    assert torch.allclose(out1.prev_sample, out2.prev_sample)


def test_step_no_nan_or_inf(scheduler_with_timesteps, dummy_batch):
    model_output, sample = dummy_batch
    t = scheduler_with_timesteps.timesteps[10]
    out = scheduler_with_timesteps.step(model_output, t, sample)
    assert torch.isfinite(out.prev_sample).all()
    assert torch.isfinite(out.pred_original_sample).all()


# ---------------------------------------------------------------------------
# 4. DMSE vs DDPM — the key behavioral difference
# ---------------------------------------------------------------------------


def test_dmse_differs_from_ddpm(dummy_batch):
    """
    DMSE omits the noise term that DDPM adds. At a mid-range timestep with
    nonzero posterior variance, the two schedulers must produce different outputs.
    At t=0 DDPM also adds no noise, so we test at a mid-range step.
    """
    dmse = DMSEScheduler()
    ddpm = DDPMScheduler()
    dmse.set_timesteps(100)
    ddpm.set_timesteps(100)

    model_output, sample = dummy_batch
    # Use a mid-range timestep where posterior variance is nonzero
    t = dmse.timesteps[50]

    torch.manual_seed(42)
    out_dmse = dmse.step(model_output, t, sample).prev_sample

    # Run DDPM several times: outputs will vary due to stochastic noise
    torch.manual_seed(42)
    out_ddpm_1 = ddpm.step(model_output, t, sample).prev_sample
    torch.manual_seed(99)
    out_ddpm_2 = ddpm.step(model_output, t, sample).prev_sample

    # DMSE is deterministic: its output must NOT match the stochastic DDPM runs
    # (unless variance happens to be exactly zero, which only occurs at t=0)
    if not torch.allclose(out_ddpm_1, out_ddpm_2, atol=1e-6):
        # DDPM is stochastic here, so DMSE must differ from at least one run
        differs = not torch.allclose(
            out_dmse, out_ddpm_1, atol=1e-6
        ) or not torch.allclose(out_dmse, out_ddpm_2, atol=1e-6)
        assert differs, "DMSE output unexpectedly matches stochastic DDPM output"


def test_dmse_matches_ddpm_posterior_mean(dummy_batch):
    """
    The DMSE step must equal the DDPM posterior mean µ_t, computed analytically.
    This is the core correctness test.
    """
    dmse = DMSEScheduler()
    dmse.set_timesteps(100)

    model_output, sample = dummy_batch
    t = int(dmse.timesteps[10])

    out = dmse.step(model_output, t, sample)

    # Reproduce µ_t manually using the same formulas as step()
    prev_t = dmse.previous_timestep(t)
    alpha_prod_t = dmse.alphas_cumprod[t]
    alpha_prod_t_prev = dmse.alphas_cumprod[prev_t] if prev_t >= 0 else dmse.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # x_0 prediction (epsilon prediction type is the default)
    pred_x0 = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    # clip_sample=True is the DDPMScheduler default
    pred_x0 = pred_x0.clamp(-1.0, 1.0)

    coeff_x0 = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
    coeff_xt = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
    expected_mu = coeff_x0 * pred_x0 + coeff_xt * sample

    assert torch.allclose(out.prev_sample, expected_mu, atol=1e-5), (
        "DMSE step does not match analytical posterior mean µ_t"
    )


# ---------------------------------------------------------------------------
# 5. Compatibility with DDPMScheduler config
# ---------------------------------------------------------------------------


def test_config_inherited(scheduler):
    """DMSEScheduler must expose the same config fields as DDPMScheduler."""
    assert hasattr(scheduler.config, "num_train_timesteps")
    assert hasattr(scheduler.config, "beta_schedule")
    assert hasattr(scheduler.config, "prediction_type")


def test_set_timesteps_multiple_values(scheduler):
    """set_timesteps should work for various step counts."""
    for n in [10, 50, 200, 1000]:
        scheduler.set_timesteps(n)
        assert len(scheduler.timesteps) == n


def test_v_prediction_type():
    """DMSEScheduler must work with v_prediction models."""
    s = DMSEScheduler(prediction_type="v_prediction")
    s.set_timesteps(50)
    B, C, H, W = 1, 1, 4, 4
    torch.manual_seed(0)
    model_output = torch.randn(B, C, H, W)
    sample = torch.randn(B, C, H, W)
    t = s.timesteps[10]
    out = s.step(model_output, t, sample)
    assert out.prev_sample.shape == sample.shape
    assert torch.isfinite(out.prev_sample).all()


def test_sample_prediction_type():
    """DMSEScheduler must work with direct sample prediction."""
    s = DMSEScheduler(prediction_type="sample")
    s.set_timesteps(50)
    B, C, H, W = 1, 1, 4, 4
    torch.manual_seed(0)
    model_output = torch.randn(B, C, H, W)
    sample = torch.randn(B, C, H, W)
    t = s.timesteps[10]
    out = s.step(model_output, t, sample)
    assert out.prev_sample.shape == sample.shape


def test_unknown_prediction_type_raises():
    """An unsupported prediction_type must raise ValueError in step()."""
    import unittest.mock

    s = DMSEScheduler(prediction_type="epsilon")
    s.set_timesteps(50)
    B, C, H, W = 1, 1, 4, 4
    model_output = torch.randn(B, C, H, W)
    sample = torch.randn(B, C, H, W)
    t = s.timesteps[10]
    # Patch via the class-level property so FrozenDict is not modified directly
    mock_cfg = unittest.mock.MagicMock()
    mock_cfg.prediction_type = "invalid"
    mock_cfg.thresholding = False
    mock_cfg.clip_sample = False
    mock_cfg.variance_type = "fixed_small"
    with unittest.mock.patch.object(
        type(s),
        "config",
        new_callable=unittest.mock.PropertyMock,
        return_value=mock_cfg,
    ):
        with pytest.raises(ValueError, match="prediction_type"):
            s.step(model_output, t, sample)

"""
Phase 4 tests: Conditioning Integration.

Tests cover the curriculum schedule, conditioning encoder wiring, regression loss,
the conditional training loop, and conditioned generation. All tests use random
tensors or tiny synthetic datasets — no data files are required.

Run via pytest:
    pytest tests/test_phase4.py -v
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset

from slicegan.networks import Generator3D, Discriminator2D
from slicegan.conditioning import ConditioningEncoder, prepare_conditioning_maps_2d
from slicegan.curriculum import get_lambda_reg, LambdaSchedule
from slicegan.model import (
    sample_conditioning_from_batch,
    _conditional_training_loop,
    generate_conditioned_samples,
    _D1, _D2, _D3,
)

# ---------------------------------------------------------------------------
# Architecture constants — tiny networks (l=8) for speed
# ---------------------------------------------------------------------------

NC = 3
NZ = 4
L = 8
LZ = 4
EMBED_DIM = 4
COND_DIM = 2 * NC   # 6

# Generator: lz=4 → l=8 (single ConvTranspose3d layer)
GK = [4]; GS = [2]; GF = [NC]; GP = [1]

# Discriminator: l=8 → 4 → 1×1
DK = [4, 4]; DS = [2, 2]; DF = [16, 16]; DP = [1, 0]

BATCH = 4
D_BATCH = L * BATCH   # 32 — matches number of fake slices


def _make_gen(embed_dim=EMBED_DIM):
    return Generator3D(nz=NZ, n_phases=NC, gk=GK, gs=GS, gf=GF, gp=GP,
                       embed_dim=embed_dim)


def _make_disc(n_conditions=COND_DIM):
    return Discriminator2D(n_phases=NC, n_conditions=n_conditions,
                           dk=DK, ds=DS, df=DF, dp=DP)


def _make_enc():
    return ConditioningEncoder(conditioning_dim=COND_DIM, embed_dim=EMBED_DIM, lz=LZ)


# Identity norm_stats — mean=0, std=1 so normalisation is a no-op
_NORM_STATS = (np.zeros(COND_DIM, dtype=np.float32),
               np.ones(COND_DIM, dtype=np.float32))


def _tiny_dataset(n=200, nc=NC, l=L):
    """200 random softmax patches [nc, l, l], same dataset for all 3 axes."""
    patches = torch.softmax(torch.randn(n, nc, l, l), dim=1)
    ds = TensorDataset(patches)
    return [ds, ds, ds]


# ---------------------------------------------------------------------------
# Curriculum tests
# ---------------------------------------------------------------------------

def test_stage1_lambda_reg_is_zero():
    """Stage 1 must always return 0.0."""
    for epoch in [1, 25, 50]:
        val = get_lambda_reg(epoch)
        assert val == 0.0, f"Expected 0.0 at epoch {epoch}, got {val}"


def test_stage2_lambda_schedule():
    """Stage 2 must ramp linearly from lambda_start to lambda_end."""
    s = LambdaSchedule(stage1_end=50, stage2_end=150, lambda_start=0.1, lambda_end=1.0)
    test_epochs = [51, 75, 100, 125, 150]
    for epoch in test_epochs:
        t = (epoch - 50) / (150 - 50)
        expected = 0.1 + t * (1.0 - 0.1)
        got = s.get(epoch)
        assert abs(got - expected) < 1e-6, \
            f"Epoch {epoch}: expected {expected:.4f}, got {got:.4f}"
    # Stage 3 must be clamped at lambda_end
    assert s.get(200) == 1.0
    assert s.get(151) == 1.0


# ---------------------------------------------------------------------------
# Architecture wiring tests
# ---------------------------------------------------------------------------

def test_generator_input_shape_with_conditioning():
    """Generator must accept z concatenated with conditioning embedding."""
    enc = _make_enc()
    G = _make_gen(embed_dim=EMBED_DIM)
    z = torch.randn(BATCH, NZ, LZ, LZ, LZ)
    cond = torch.randn(BATCH, COND_DIM)
    with torch.no_grad():
        emb = enc(cond)                         # [BATCH, EMBED_DIM, LZ, LZ, LZ]
        gen_input = torch.cat([z, emb], dim=1)  # [BATCH, NZ+EMBED_DIM, LZ, LZ, LZ]
        out = G(gen_input)
    assert gen_input.shape == (BATCH, NZ + EMBED_DIM, LZ, LZ, LZ)
    assert out.shape == (BATCH, NC, L, L, L), f"Got {out.shape}"


def test_discriminator_input_channels():
    """Discriminator must accept n_phases + conditioning_dim input channels."""
    D = _make_disc(n_conditions=COND_DIM)
    slices = torch.randn(BATCH, NC, L, L)
    cond = torch.randn(BATCH, COND_DIM)
    cond_maps = prepare_conditioning_maps_2d(cond, spatial_size=L)   # [B, 6, L, L]
    x = torch.cat([slices, cond_maps], dim=1)                        # [B, NC+6, L, L]
    assert x.shape[1] == NC + COND_DIM
    with torch.no_grad():
        score, reg = D(x)
    assert score.shape == (BATCH, 1)
    assert reg.shape == (BATCH, 2 * NC)


# ---------------------------------------------------------------------------
# Regression loss tests
# ---------------------------------------------------------------------------

def test_regression_loss_zero_on_perfect():
    """MSE loss must be zero when prediction equals target exactly."""
    pred = torch.rand(BATCH, COND_DIM)
    target = pred.clone()
    loss = F.mse_loss(pred, target)
    assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"


def test_regression_loss_decreases():
    """Discriminator regression head must improve when trained on fixed conditioning."""
    torch.manual_seed(0)
    D = _make_disc(n_conditions=COND_DIM)
    opt = Adam(D.parameters(), lr=1e-3)

    # Constant target cond vector for every batch
    target_cond = torch.rand(D_BATCH, COND_DIM)
    cond_maps = prepare_conditioning_maps_2d(target_cond, spatial_size=L)

    losses = []
    for step in range(220):
        slices = torch.softmax(torch.randn(D_BATCH, NC, L, L), dim=1)
        x = torch.cat([slices, cond_maps], dim=1)
        opt.zero_grad()
        _, reg = D(x)
        loss = F.mse_loss(reg, target_cond)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Regression loss did not decrease: initial={losses[0]:.4f}, "
        f"final={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Conditioning normalisation
# ---------------------------------------------------------------------------

def test_conditioning_normalisation():
    """sample_conditioning_from_batch must z-score correctly using norm_stats."""
    torch.manual_seed(1)
    nc, l = 3, 8
    batch = 32

    # Build patches where VF is exactly 1/nc (uniform)
    # and PS will be some positive value — we only check VF half
    patches = torch.zeros(batch, nc, l, l)
    for c in range(nc):
        patches[:, c, :l // nc * c:l // nc * (c + 1), :] = 1.0
    # Actually create one-hot patches with known VF
    idx = torch.randint(0, nc, (batch, l, l))
    patches = torch.zeros(batch, nc, l, l)
    patches.scatter_(1, idx.unsqueeze(1), 1.0)

    raw = sample_conditioning_from_batch(patches, _NORM_STATS, torch.device('cpu'))
    # With identity norm_stats the output equals the raw conditioning vector
    assert raw.shape == (batch, 2 * nc)

    # Now test with non-trivial norm_stats: mean=raw.mean(0), std=raw.std(0)
    raw_np = raw.numpy()
    mean_np = raw_np.mean(axis=0)
    std_np = raw_np.std(axis=0) + 1e-8
    norm_stats_nontrivial = (mean_np.astype(np.float32), std_np.astype(np.float32))

    normed = sample_conditioning_from_batch(patches, norm_stats_nontrivial, torch.device('cpu'))
    # Each column should be approximately zero-mean
    col_means = normed.mean(dim=0).abs()
    assert col_means.max().item() < 0.3, \
        f"Column means not near zero after normalisation: {col_means.tolist()}"


# ---------------------------------------------------------------------------
# Conditioned generation shape
# ---------------------------------------------------------------------------

def test_conditioned_generation_shape():
    """generate_conditioned_samples must return [n_samples, n_phases, l, l, l]."""
    # Use full-size architecture for this test (l=64)
    nz_full = 64
    n_phases = 3
    embed_dim_full = 32
    cond_dim_full = 2 * n_phases

    GK_F = [4, 4, 4, 4, 4]
    GS_F = [2, 2, 2, 2, 2]
    GF_F = [512, 256, 128, 64, n_phases]
    GP_F = [2, 2, 2, 2, 3]

    netG = Generator3D(nz=nz_full, n_phases=n_phases,
                       gk=GK_F, gs=GS_F, gf=GF_F, gp=GP_F,
                       embed_dim=embed_dim_full)
    cond_enc = ConditioningEncoder(conditioning_dim=cond_dim_full,
                                   embed_dim=embed_dim_full, lz=4)

    cond_vector = [0.33, 0.33, 0.34, 2.5, 3.0, 1.5]  # raw unnormalised
    norm_stats = (
        np.zeros(cond_dim_full, dtype=np.float32),
        np.ones(cond_dim_full, dtype=np.float32),
    )
    n_samples = 3
    out = generate_conditioned_samples(
        netG, cond_enc, cond_vector, n_samples=n_samples,
        nz=nz_full, lz=4, norm_stats=norm_stats, device='cpu',
    )
    assert out.shape == (n_samples, n_phases, 64, 64, 64), f"Got {out.shape}"
    # Soft softmax output: values in [0, 1], channels sum to 1
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


# ---------------------------------------------------------------------------
# No NaN losses over 500 iterations
# ---------------------------------------------------------------------------

def test_no_nan_losses():
    """Conditional training loop must be numerically stable over 500 inner steps."""
    torch.manual_seed(42)
    datasets = _tiny_dataset(n=600)
    netG = _make_gen()
    netD = _make_disc()
    enc = _make_enc()

    result = _conditional_training_loop(
        datasets, netG, netD, enc,
        nc=NC, l=L, nz=NZ, pth='',
        norm_stats=_NORM_STATS,
        isotropic=True,
        num_epochs=500,   # each epoch = 1 step with 600 patches / D_BATCH=32
        batch_size=BATCH,
        D_batch_size=D_BATCH,
        lrg=1e-4, lrd=1e-4,
        Lambda=10, critic_iters=5, lz=LZ, workers=0,
    )

    # Trim to first 500 entries across all log lists
    for key in ('disc_real_log', 'disc_fake_log', 'gp_log', 'Wass_log'):
        log = result[key][:500]
        nan_count = sum(math.isnan(v) for v in log)
        assert nan_count == 0, f"{key} has {nan_count} NaN values"


# ---------------------------------------------------------------------------
# Rough conditioning accuracy
# ---------------------------------------------------------------------------

def test_conditioning_rough_accuracy():
    """After short training the generator should approximate the target VF."""
    torch.manual_seed(7)
    nc, l, lz = NC, L, LZ
    nz = NZ
    embed_dim = EMBED_DIM

    # Build a dataset where every patch has exactly uniform VF = 1/nc
    n_patches = 400
    idx = torch.randint(0, nc, (n_patches, l, l))
    patches = torch.zeros(n_patches, nc, l, l)
    patches.scatter_(1, idx.unsqueeze(1), 1.0)
    # Force exactly equal VF distribution
    per_class = n_patches // nc
    for c in range(nc):
        block = torch.zeros(per_class, nc, l, l)
        block[:, c, :, :] = 1.0
        patches[c * per_class:(c + 1) * per_class] = block

    ds = TensorDataset(patches)
    datasets = [ds, ds, ds]

    # Training norm_stats: VF = 1/nc for all phases, PS varies
    # Use identity norm for simplicity
    norm_stats = (np.zeros(2 * nc, dtype=np.float32),
                  np.ones(2 * nc, dtype=np.float32))

    netG = Generator3D(nz=nz, n_phases=nc, gk=GK, gs=GS, gf=GF, gp=GP,
                       embed_dim=embed_dim)
    netD = Discriminator2D(n_phases=nc, n_conditions=COND_DIM,
                           dk=DK, ds=DS, df=DF, dp=DP)
    enc = ConditioningEncoder(conditioning_dim=COND_DIM, embed_dim=embed_dim, lz=lz)

    _conditional_training_loop(
        datasets, netG, netD, enc,
        nc=nc, l=l, nz=nz, pth='',
        norm_stats=norm_stats,
        isotropic=True,
        num_epochs=300,
        batch_size=BATCH,
        D_batch_size=D_BATCH,
        lrg=1e-3, lrd=1e-3,
        Lambda=10, critic_iters=5, lz=lz, workers=0,
    )

    # Target: uniform VF [1/3, 1/3, 1/3], PS ~ 0 (binary patches)
    target_vf = 1.0 / nc
    cond_vector = [target_vf] * nc + [0.0] * nc  # raw, unnormalised

    generated = generate_conditioned_samples(
        netG, enc, cond_vector, n_samples=20,
        nz=nz, lz=lz, norm_stats=norm_stats, device='cpu',
    )  # [20, nc, l, l, l] — soft softmax output

    # Compute per-phase VF over all generated volumes via soft probabilities
    mean_vf = generated.mean(dim=[0, 2, 3, 4])  # [nc]

    for c in range(nc):
        rel_err = abs(mean_vf[c].item() - target_vf) / (target_vf + 1e-8)
        assert rel_err < 0.20, (
            f"Phase {c}: target VF={target_vf:.3f}, got {mean_vf[c]:.3f}, "
            f"relative error={rel_err:.3f} > 0.20"
        )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

def main():
    raise SystemExit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()

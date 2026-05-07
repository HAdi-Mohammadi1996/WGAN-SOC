"""
Phase 2 tests: Model Architecture.

Tests cover Generator3D, Discriminator2D, ConditioningEncoder, and the
slicegan_nets factory function. All tests use random tensors — no data files
are required.

Run via pytest:
    pytest tests/test_phase2.py -v
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch
import torch.nn as nn

from slicegan.networks import Generator3D, Discriminator2D, slicegan_nets
from slicegan.conditioning import ConditioningEncoder, prepare_conditioning_maps_2d


# ---------------------------------------------------------------------------
# Default architecture constants (SliceGAN Table 1)
# ---------------------------------------------------------------------------

N_PHASES = 3
NZ = 64
LZ = 4
EMBED_DIM = 32
CONDITIONING_DIM = 2 * N_PHASES  # 6: [vf_1..vf_3, ps_1..ps_3]

GK = [4, 4, 4, 4, 4]
GS = [2, 2, 2, 2, 2]
GF = [512, 256, 128, 64, N_PHASES]
GP = [2, 2, 2, 2, 3]

DK = [4, 4, 4, 4, 4]
DS = [2, 2, 2, 2, 2]
DF = [64, 128, 256, 512, 512]  # last entry is feature dim for linear heads
DP = [1, 1, 1, 1, 0]

BATCH = 4


def _make_generator(embed_dim=0):
    return Generator3D(nz=NZ, n_phases=N_PHASES, gk=GK, gs=GS, gf=GF, gp=GP,
                       embed_dim=embed_dim)


def _make_discriminator(n_conditions=0):
    return Discriminator2D(n_phases=N_PHASES, n_conditions=n_conditions,
                           dk=DK, ds=DS, df=DF, dp=DP)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

def test_generator_output_shape():
    G = _make_generator()
    z = torch.randn(BATCH, NZ, LZ, LZ, LZ)
    with torch.no_grad():
        out = G(z)
    assert out.shape == (BATCH, N_PHASES, 64, 64, 64), f"Got {out.shape}"


def test_softmax_valid():
    """Phase channels must sum to 1 at every voxel."""
    G = _make_generator()
    G.eval()
    z = torch.randn(BATCH, NZ, LZ, LZ, LZ)
    with torch.no_grad():
        out = G(z)
    phase_sum = out.sum(dim=1)  # [batch, 64, 64, 64]
    max_dev = (phase_sum - 1.0).abs().max().item()
    assert max_dev < 1e-5, f"Max softmax deviation: {max_dev}"


# ---------------------------------------------------------------------------
# Discriminator tests
# ---------------------------------------------------------------------------

def test_discriminator_critic_shape():
    D = _make_discriminator()
    x = torch.randn(BATCH, N_PHASES, 64, 64)
    with torch.no_grad():
        score, _ = D(x)
    assert score.shape == (BATCH, 1), f"Got {score.shape}"


def test_discriminator_regression_shape():
    D = _make_discriminator()
    x = torch.randn(BATCH, N_PHASES, 64, 64)
    with torch.no_grad():
        _, reg = D(x)
    assert reg.shape == (BATCH, 2 * N_PHASES), f"Got {reg.shape}"


def test_regression_output_range():
    """Regression output must be in [0, 1] due to Sigmoid."""
    D = _make_discriminator()
    x = torch.randn(BATCH, N_PHASES, 64, 64)
    with torch.no_grad():
        _, reg = D(x)
    assert reg.min().item() >= 0.0
    assert reg.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Uniform information density
# ---------------------------------------------------------------------------

def test_uniform_info_density():
    """Averaged over many random inputs, edge and centre output magnitudes
    must be within 10% — confirming the architecture has no edge artifacts."""
    G = _make_generator()
    G.eval()
    torch.manual_seed(0)
    n_samples = 128
    with torch.no_grad():
        z = torch.randn(n_samples, NZ, LZ, LZ, LZ)
        out = G(z)  # [n_samples, N_PHASES, 64, 64, 64]
    # Average over batch and phase → spatial mean map [64, 64, 64]
    spatial_mean = out.mean(dim=[0, 1])
    # Edge: outer 4 voxels on each face
    edge_vals = torch.cat([
        spatial_mean[:4, :, :].reshape(-1),
        spatial_mean[-4:, :, :].reshape(-1),
        spatial_mean[:, :4, :].reshape(-1),
        spatial_mean[:, -4:, :].reshape(-1),
        spatial_mean[:, :, :4].reshape(-1),
        spatial_mean[:, :, -4:].reshape(-1),
    ])
    center_vals = spatial_mean[4:-4, 4:-4, 4:-4].reshape(-1)
    edge_mean = edge_vals.mean().item()
    center_mean = center_vals.mean().item()
    ratio = abs(edge_mean - center_mean) / (abs(center_mean) + 1e-8)
    assert ratio < 0.10, (
        f"Edge/centre ratio {ratio:.4f} exceeds 10%; "
        "check uniform information density parameters."
    )


# ---------------------------------------------------------------------------
# Conditioning encoder tests
# ---------------------------------------------------------------------------

def test_conditioning_encoder_shape():
    enc = ConditioningEncoder(conditioning_dim=CONDITIONING_DIM,
                              embed_dim=EMBED_DIM, lz=LZ)
    cond = torch.randn(BATCH, CONDITIONING_DIM)
    with torch.no_grad():
        out = enc(cond)
    assert out.shape == (BATCH, EMBED_DIM, LZ, LZ, LZ), f"Got {out.shape}"


def test_generator_with_conditioning():
    """Generator must accept z concatenated with conditioning embedding."""
    enc = ConditioningEncoder(conditioning_dim=CONDITIONING_DIM,
                              embed_dim=EMBED_DIM, lz=LZ)
    G = _make_generator(embed_dim=EMBED_DIM)
    cond = torch.randn(BATCH, CONDITIONING_DIM)
    z = torch.randn(BATCH, NZ, LZ, LZ, LZ)
    with torch.no_grad():
        emb = enc(cond)                         # [batch, embed_dim, 4, 4, 4]
        gen_input = torch.cat([z, emb], dim=1)  # [batch, nz+embed_dim, 4, 4, 4]
        out = G(gen_input)
    assert out.shape == (BATCH, N_PHASES, 64, 64, 64), f"Got {out.shape}"


def test_discriminator_with_conditioning():
    """Discriminator must accept slices with conditioning channels appended."""
    D = _make_discriminator(n_conditions=CONDITIONING_DIM)
    cond = torch.randn(BATCH, CONDITIONING_DIM)
    cond_maps = prepare_conditioning_maps_2d(cond, spatial_size=64)  # [B, 6, 64, 64]
    slices = torch.randn(BATCH, N_PHASES, 64, 64)
    x = torch.cat([slices, cond_maps], dim=1)  # [B, N_PHASES+CONDITIONING_DIM, 64, 64]
    with torch.no_grad():
        score, reg = D(x)
    assert score.shape == (BATCH, 1)
    assert reg.shape == (BATCH, 2 * N_PHASES)


# ---------------------------------------------------------------------------
# Parameter count test
# ---------------------------------------------------------------------------

def test_parameter_counts():
    """Parameter counts should print without error."""
    G = _make_generator(embed_dim=EMBED_DIM)
    D = _make_discriminator(n_conditions=CONDITIONING_DIM)
    enc = ConditioningEncoder(conditioning_dim=CONDITIONING_DIM,
                              embed_dim=EMBED_DIM, lz=LZ)
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    e_params = sum(p.numel() for p in enc.parameters())
    print(f"Generator params:    {g_params:,}")
    print(f"Discriminator params:{d_params:,}")
    print(f"Encoder params:      {e_params:,}")
    assert g_params > 0
    assert d_params > 0
    assert e_params > 0


# ---------------------------------------------------------------------------
# Large volume inference
# ---------------------------------------------------------------------------

def _compute_output_size(lz, gk, gs, gp):
    """Compute generator spatial output size for a given input spatial size."""
    size = lz
    for k, s, p in zip(gk, gs, gp):
        size = (size - 1) * s - 2 * p + k
    return size


def test_large_volume_inference():
    """Generator must scale to larger input spatial size at inference."""
    lz_large = 8
    expected_size = _compute_output_size(lz_large, GK, GS, GP)
    assert expected_size > 64, "Large input should produce larger output than training size"

    G = _make_generator()
    G.eval()
    z = torch.randn(2, NZ, lz_large, lz_large, lz_large)
    with torch.no_grad():
        out = G(z)
    assert out.shape == (2, N_PHASES, expected_size, expected_size, expected_size), \
        f"Expected {(2, N_PHASES, expected_size, expected_size, expected_size)}, got {out.shape}"


# ---------------------------------------------------------------------------
# slicegan_nets factory
# ---------------------------------------------------------------------------

def test_slicegan_nets_compatibility():
    """slicegan_nets factory must return generator and discriminator with correct shapes."""
    netG, netD = slicegan_nets(
        path="dummy", training=True, imtype="nphase",
        dk=DK, ds=DS, df=DF, dp=DP,
        gk=GK, gs=GS, gf=GF, gp=GP,
        nz=NZ, n_phases=N_PHASES, n_conditions=0, embed_dim=0,
    )
    z = torch.randn(BATCH, NZ, LZ, LZ, LZ)
    x = torch.randn(BATCH, N_PHASES, 64, 64)
    with torch.no_grad():
        gen_out = netG(z)
        score, reg = netD(x)
    assert gen_out.shape == (BATCH, N_PHASES, 64, 64, 64)
    assert score.shape == (BATCH, 1)
    assert reg.shape == (BATCH, 2 * N_PHASES)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

def main():
    raise SystemExit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()

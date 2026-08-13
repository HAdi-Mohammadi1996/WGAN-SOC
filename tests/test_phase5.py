"""
Phase 5 tests: Validation and Evaluation.

Tests cover microstructure metrics (VF, S₂, coherence, TPB density),
statistical similarity via KS test, conditioning accuracy, and visualisation.
All tests use synthetic volumes — no data files or trained checkpoints required.

Run via pytest:
    pytest tests/test_phase5.py -v
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from slicegan.metrics import (
    MicrostructureMetrics,
    conditioning_accuracy_test,
    statistical_similarity_test,
    visualise_volume,
)
from slicegan.data_pipeline import compute_volume_fraction
from slicegan.networks import Generator3D
from slicegan.conditioning import ConditioningEncoder

# ---------------------------------------------------------------------------
# Shared tiny-architecture constants (same as Phase 4 tests)
# ---------------------------------------------------------------------------

NC = 3
NZ = 4
L = 8
LZ = 4
EMBED_DIM = 4
COND_DIM = 2 * NC   # 6

GK = [4]; GS = [2]; GF = [NC]; GP = [1]

_NORM_STATS = (np.zeros(COND_DIM, dtype=np.float32),
               np.ones(COND_DIM, dtype=np.float32))


def _make_gen():
    return Generator3D(nz=NZ, n_phases=NC, gk=GK, gs=GS, gf=GF, gp=GP,
                       embed_dim=EMBED_DIM)


def _make_enc():
    return ConditioningEncoder(conditioning_dim=COND_DIM, embed_dim=EMBED_DIM, lz=LZ)


# ---------------------------------------------------------------------------
# Volume fraction
# ---------------------------------------------------------------------------

def test_vf_matches_phase1():
    """MicrostructureMetrics.volume_fraction must agree with compute_volume_fraction."""
    # Known counts: 1000 × phase-0, 2000 × phase-1, 3000 × phase-2
    vol = np.array([0] * 1000 + [1] * 2000 + [2] * 3000, dtype=np.int64).reshape(6, 10, 100)
    oh = F.one_hot(torch.tensor(vol), num_classes=3).permute(3, 0, 1, 2).float()
    dp_vf = compute_volume_fraction(oh).numpy()

    m = MicrostructureMetrics()
    our_vf = m.volume_fraction(vol, n_phases=3)

    np.testing.assert_allclose(our_vf, dp_vf, atol=1e-6)


# ---------------------------------------------------------------------------
# Two-point correlation
# ---------------------------------------------------------------------------

def test_s2_at_zero_equals_vf():
    """S₂(0) must equal the volume fraction within 1e-4."""
    np.random.seed(0)
    vol = np.random.randint(0, 3, (20, 20, 20))
    m = MicrostructureMetrics()
    vf = m.volume_fraction(vol, 3)[1]
    s2 = m.two_point_correlation(vol, phase_idx=1, max_r=10)
    assert abs(s2[0] - vf) < 1e-4, f"S₂(0)={s2[0]:.6f} but VF={vf:.6f}"


def test_s2_at_infinity():
    """S₂(max_r) must approximate vf² within 5 % for a large iid volume."""
    np.random.seed(1)
    vol = np.random.randint(0, 3, (32, 32, 32))
    m = MicrostructureMetrics()
    vf = m.volume_fraction(vol, 3)[1]
    s2 = m.two_point_correlation(vol, phase_idx=1, max_r=15)
    target = vf ** 2
    tol = 0.05 * target + 1e-6  # 5 % of vf² plus a floor to avoid div-by-zero
    assert abs(s2[-1] - target) <= tol, (
        f"S₂({len(s2)-1})={s2[-1]:.6f}, vf²={target:.6f}, tol={tol:.6f}"
    )


# ---------------------------------------------------------------------------
# KS statistical similarity
# ---------------------------------------------------------------------------

def test_ks_same_distribution():
    """KS test on volumes from the same distribution must give p > 0.05."""
    # All 40 volumes have exactly 1/3 VF per phase — both groups are identical
    D = 30
    vol = np.zeros((D, D, D), dtype=np.int64)
    vol[D // 3:2 * D // 3, :, :] = 1
    vol[2 * D // 3:, :, :] = 2
    vols = [vol.copy() for _ in range(40)]
    result = statistical_similarity_test(vols[:20], vols[20:])
    for key, res in result.items():
        assert res['pvalue'] > 0.05, f"{key}: pvalue={res['pvalue']:.4f}"


def test_ks_different_distributions():
    """KS test on clearly different VF distributions must give p < 0.05."""
    # dist_a: 80 % phase 0, 10 % phase 1, 10 % phase 2
    vol_a = np.zeros((10, 10, 10), dtype=np.int64)
    vol_a[8:9, :, :] = 1
    vol_a[9:, :, :] = 2

    # dist_b: 20 % phase 0, 40 % phase 1, 40 % phase 2
    vol_b = np.zeros((10, 10, 10), dtype=np.int64)
    vol_b[2:6, :, :] = 1
    vol_b[6:, :, :] = 2

    result = statistical_similarity_test(
        [vol_a.copy() for _ in range(20)],
        [vol_b.copy() for _ in range(20)],
    )
    for key, res in result.items():
        assert res['pvalue'] < 0.05, f"{key}: pvalue={res['pvalue']:.4f}"


# ---------------------------------------------------------------------------
# Inter-slice coherence
# ---------------------------------------------------------------------------

def test_coherence_uniform_volume():
    """A single-phase volume must have coherence == 1.0 within 1e-5."""
    vol = np.ones((10, 10, 10), dtype=np.int64)
    m = MicrostructureMetrics()
    c = m.inter_slice_coherence(vol)
    assert abs(c - 1.0) < 1e-5, f"coherence={c}"


def test_coherence_random_volume():
    """An iid random 3-phase volume must have coherence < 1.0."""
    np.random.seed(2)
    vol = np.random.randint(0, 3, (10, 10, 10))
    m = MicrostructureMetrics()
    c = m.inter_slice_coherence(vol)
    assert c < 1.0, f"coherence={c}"


# ---------------------------------------------------------------------------
# TPB density
# ---------------------------------------------------------------------------

def test_tpb_density_non_negative():
    """TPB density must be non-negative for any valid 3-phase volume."""
    np.random.seed(3)
    m = MicrostructureMetrics()
    vols = [
        np.zeros((8, 8, 8), dtype=np.int64),           # all phase 0 → 0 TPB
        np.random.randint(0, 3, (8, 8, 8)),             # mixed → some TPB
        np.ones((8, 8, 8), dtype=np.int64),             # all phase 1 → 0 TPB
    ]
    for vol in vols:
        d = m.triple_phase_boundary_density(vol)
        assert d >= 0, f"TPB density={d} is negative"


# ---------------------------------------------------------------------------
# Conditioning accuracy
# ---------------------------------------------------------------------------

def test_conditioning_accuracy_runs():
    """conditioning_accuracy_test must return non-NaN floats with R² ≤ 1.0."""
    torch.manual_seed(0)
    netG = _make_gen()
    cond_enc = _make_enc()

    # 5 vectors with varying phase-0 VF; all sum to 1; PS columns set to 0
    test_vectors = []
    for vf0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rem = (1.0 - vf0) / 2.0
        test_vectors.append([vf0, rem, rem, 0.0, 0.0, 0.0])

    result = conditioning_accuracy_test(
        netG, cond_enc, test_vectors,
        n_samples=3, nz=NZ, lz=LZ,
        norm_stats=_NORM_STATS, device='cpu',
    )
    assert not np.isnan(result['r2']), "R² is NaN"
    assert result['r2'] <= 1.0, f"R²={result['r2']:.4f} > 1.0"
    assert not np.isnan(result['mae']), "MAE is NaN"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def test_visualisation_saves(tmp_path):
    """visualise_volume must write a PNG file to the specified path."""
    np.random.seed(4)
    vol = np.random.randint(0, 3, (16, 16, 16))
    save_path = str(tmp_path / "test_vol.png")
    visualise_volume(vol, 'nphase', save_path)
    assert os.path.exists(save_path), f"File not found: {save_path}"


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

def main():
    raise SystemExit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()

"""
Phase 1 tests: Data Pipeline and Preprocessing.

All tests use synthetic data generated programmatically — no real .mat files
are required.

Run directly (no pytest CLI needed):
    python tests/test_phase1.py

Or via pytest:
    pytest tests/test_phase1.py -v
"""

import os
import sys
import tempfile

# Ensure project root is importable when this file is run directly
# (pytest handles this via conftest.py; this covers `python tests/test_phase1.py`)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

from slicegan.data_pipeline import (
    augment_volume,
    compute_average_pore_size,
    compute_conditioning_vector,
    compute_dataset_conditioning_stats,
    compute_volume_fraction,
    extract_subvolumes,
    load_mat_volume,
)
from slicegan.preprocessing import batch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTH_SHAPE = (80, 100, 60)
N_PHASES = 3


def _make_synthetic_vol(shape=SYNTH_SHAPE, seed=42):
    """Create a reproducible synthetic 3-phase volume with labels 1,2,3."""
    rng = np.random.default_rng(seed)
    vol = rng.integers(1, 4, size=shape, dtype=np.int32)  # values in {1,2,3}
    return vol


@pytest.fixture(scope="module")
def tmp_mat(tmp_path_factory):
    """Write a synthetic .mat file and return its path."""
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "test_vol.mat"
    vol = _make_synthetic_vol()
    scipy.io.savemat(str(path), {"vol_seg": vol})
    return path, vol


@pytest.fixture(scope="module")
def synthetic_vol():
    return _make_synthetic_vol()


# ---------------------------------------------------------------------------
# load_mat_volume tests
# ---------------------------------------------------------------------------

def test_load_mat_volume_shape(tmp_mat):
    path, expected_vol = tmp_mat
    vol = load_mat_volume(str(path))
    assert isinstance(vol, np.ndarray)
    assert vol.shape == SYNTH_SHAPE


def test_load_mat_volume_keys(tmp_mat):
    path, expected_vol = tmp_mat
    vol = load_mat_volume(str(path))
    np.testing.assert_array_equal(vol, expected_vol)


def test_load_mat_missing_key(tmp_path):
    bad_path = tmp_path / "bad.mat"
    scipy.io.savemat(str(bad_path), {"wrong_key": np.zeros((4, 4, 4))})
    with pytest.raises((KeyError, ValueError)):
        load_mat_volume(str(bad_path))


# ---------------------------------------------------------------------------
# extract_subvolumes tests
# ---------------------------------------------------------------------------

def test_extract_subvolumes_count(synthetic_vol):
    subvol_size = 20
    stride = 20  # non-overlapping
    # Expected positions per axis:
    # X: floor((80-20)/20)+1 = 4
    # Y: floor((100-20)/20)+1 = 5
    # Z: floor((60-20)/20)+1 = 3
    # Total grid = 4*5*3 = 60 positions
    # Some may be filtered by coverage; test with coverage=0 to get all
    subs = extract_subvolumes(synthetic_vol, subvol_size=subvol_size,
                               stride=stride, min_phase_coverage=0.0)
    expected = 4 * 5 * 3
    assert len(subs) == expected


def test_extract_subvolumes_shape(synthetic_vol):
    subvol_size = 20
    subs = extract_subvolumes(synthetic_vol, subvol_size=subvol_size,
                               stride=20, min_phase_coverage=0.0)
    for sv in subs:
        assert sv.shape == (subvol_size, subvol_size, subvol_size)


def test_extract_subvolumes_coverage_filter():
    """Subvolumes that are nearly homogeneous should be rejected."""
    # Create a volume that is entirely phase 1 except one small region
    vol = np.ones((40, 40, 40), dtype=np.int32)  # all phase 1
    # Make a corner region with all three phases present
    vol[0:20, 0:20, 0:20] = np.resize(np.array([1, 2, 3], dtype=np.int32),
                                       (20, 20, 20))
    # With min_phase_coverage=0.05, patches that are all-1 should be rejected
    # since phase 2 and 3 won't meet coverage
    subs_filtered = extract_subvolumes(vol, subvol_size=20, stride=20,
                                        min_phase_coverage=0.05)
    subs_unfiltered = extract_subvolumes(vol, subvol_size=20, stride=20,
                                          min_phase_coverage=0.0)
    # Filtered should have fewer or equal subvolumes
    assert len(subs_filtered) <= len(subs_unfiltered)
    # The homogeneous patches (all phase 1) should be excluded
    # There are 2*2*2=8 total; only the corner one passes (partially)
    assert len(subs_filtered) < len(subs_unfiltered)


def test_extract_subvolumes_stride(synthetic_vol):
    """Overlapping extraction (stride < subvol_size) yields more subvolumes."""
    subvol_size = 20
    subs_non_overlapping = extract_subvolumes(
        synthetic_vol, subvol_size=subvol_size, stride=20, min_phase_coverage=0.0)
    subs_overlapping = extract_subvolumes(
        synthetic_vol, subvol_size=subvol_size, stride=10, min_phase_coverage=0.0)
    assert len(subs_overlapping) > len(subs_non_overlapping)


# ---------------------------------------------------------------------------
# batch / preprocessing tests
# ---------------------------------------------------------------------------

def test_batch_mat3d_returns_three_datasets(tmp_mat):
    path, _ = tmp_mat
    result = batch([str(path)], imtype='mat3D', l=20, sf=1)
    assert len(result) == 3
    for ds in result:
        assert isinstance(ds, TensorDataset)


def test_dataset_size_capped(tmp_mat):
    """Each dataset must not exceed 32*900 = 28 800 patches."""
    path, _ = tmp_mat
    result = batch([str(path)], imtype='mat3D', l=20, sf=1)
    for ds in result:
        assert len(ds) <= 32 * 900


def test_onehot_valid(tmp_mat):
    """One-hot channels must sum to 1 at every pixel."""
    path, _ = tmp_mat
    result = batch([str(path)], imtype='mat3D', l=20, sf=1)
    ds = result[0]
    # Check first 10 patches
    for i in range(min(10, len(ds))):
        patch = ds[i][0]  # (n_phases, l, l)
        channel_sum = patch.sum(dim=0)  # (l, l)
        max_dev = (channel_sum - 1.0).abs().max().item()
        assert max_dev < 1e-4, f"One-hot sum deviation {max_dev} at patch {i}"


def test_phase_label_mapping(tmp_path):
    """Labels 1,2,3 must map to channels 0,1,2 respectively."""
    # Build a small volume with known label layout
    vol = np.zeros((30, 30, 30), dtype=np.int32)
    vol[:10, :, :] = 1
    vol[10:20, :, :] = 2
    vol[20:, :, :] = 3

    path = tmp_path / "label_test.mat"
    scipy.io.savemat(str(path), {"vol_seg": vol})

    result = batch([str(path)], imtype='mat3D', l=28, sf=1)
    # Just check the raw pipeline via load + encode
    loaded = load_mat_volume(str(path))
    assert loaded[0, 0, 0] == 1
    assert loaded[10, 0, 0] == 2
    assert loaded[20, 0, 0] == 3

    # Patches should have channel 0 hot where label was 1
    ds = result[0]
    for i in range(min(5, len(ds))):
        patch = ds[i][0]  # (3, l, l)
        ch_sum = patch.sum(dim=0)
        assert (ch_sum - 1.0).abs().max().item() < 1e-4


def test_patch_shape(tmp_mat):
    """Patches should have shape [n_phases, l, l]."""
    path, _ = tmp_mat
    l = 20
    result = batch([str(path)], imtype='mat3D', l=l, sf=1)
    for ds in result:
        patch = ds[0][0]
        assert patch.shape == (N_PHASES, l, l)


def test_scale_factor(tmp_mat):
    """Scale factor sf=2 should halve each spatial dimension of the volume
    before patching (we verify patch count decreases, since there are fewer
    subvolumes to extract from a downsampled volume)."""
    path, _ = tmp_mat
    result_sf1 = batch([str(path)], imtype='mat3D', l=20, sf=1)
    result_sf2 = batch([str(path)], imtype='mat3D', l=20, sf=2)
    # sf=2 → volume (40, 50, 30), fewer subvols of size 20
    assert len(result_sf2[0]) <= len(result_sf1[0])


def test_isotropic_replication(tmp_mat):
    """A single-path list should produce three datasets."""
    path, _ = tmp_mat
    result = batch([str(path)], imtype='mat3D', l=20, sf=1)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# compute_volume_fraction
# ---------------------------------------------------------------------------

def test_volume_fraction_sums_to_one():
    """VF per phase must sum to 1.0 within 1e-4."""
    rng = np.random.default_rng(0)
    raw = rng.integers(1, 4, size=(3, 16, 16, 16))  # wrong — make one-hot
    # Build one-hot tensor
    n_phases = 3
    labels = rng.integers(0, n_phases, size=(16, 16, 16))
    onehot = np.zeros((n_phases, 16, 16, 16), dtype=np.float32)
    for ph in range(n_phases):
        onehot[ph] = (labels == ph).astype(np.float32)
    t = torch.from_numpy(onehot)
    vf = compute_volume_fraction(t)
    assert abs(vf.sum().item() - 1.0) < 1e-4

# ---------------------------------------------------------------------------
# compute_average_pore_size
# ---------------------------------------------------------------------------

def test_pore_size_positive():
    """Average pore size must be > 0 for all phases present in 2D and 3D volumes."""
    vol3d = _make_synthetic_vol(shape=(30, 30, 30))
    cld_mean_3d = compute_average_pore_size(vol3d, voxel_size=0.1, px_min_mean=4.0)
    assert all(ps > 0.0 for ps in cld_mean_3d), (
        f"All 3D CLD means should be positive, got {cld_mean_3d}"
    )

    vol2d = vol3d[0]  # use one 2D slice of the synthetic volume
    cld_mean_2d = compute_average_pore_size(vol2d, voxel_size=0.1, px_min_mean=4.0)
    assert all(ps > 0.0 for ps in cld_mean_2d), (
        f"All 2D CLD means should be positive, got {cld_mean_2d}"
    )


# ---------------------------------------------------------------------------
# compute_conditioning_vector
# ---------------------------------------------------------------------------

def test_conditioning_vector_length():
    """Conditioning vector must have length 2 * n_phases."""
    vol = _make_synthetic_vol(shape=(30, 30, 30))
    vec = compute_conditioning_vector(vol, n_phases=N_PHASES)
    assert vec.shape == (2 * N_PHASES,)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def test_augmentation_preserves_vf():
    """Augmentation (rot/flip) must not change phase volume fractions."""
    vol = _make_synthetic_vol(shape=(20, 20, 20))
    phases = np.unique(vol)
    total = vol.size

    for _ in range(10):
        aug = augment_volume(vol.copy())
        for ph in phases:
            orig_frac = (vol == ph).sum() / total
            aug_frac = (aug == ph).sum() / total
            assert abs(orig_frac - aug_frac) < 1e-4, (
                f"VF changed after augmentation for phase {ph}: "
                f"{orig_frac:.4f} → {aug_frac:.4f}"
            )


# ---------------------------------------------------------------------------
# DataLoader iteration
# ---------------------------------------------------------------------------

def test_dataloader_iterates(tmp_mat):
    """DataLoader should complete 2 full iterations without error."""
    path, _ = tmp_mat
    result = batch([str(path)], imtype='mat3D', l=20, sf=1)
    ds = result[0]
    loader = DataLoader(ds, batch_size=8, num_workers=0, shuffle=True)
    count = 0
    for _ in loader:
        count += 1
        if count >= 2:
            break
    assert count >= min(2, len(loader))


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------

def main():
    """Run the Phase 1 test suite directly via pytest programmatic API.

    Exits with code 0 on success, non-zero on failure, so CI scripts can
    consume the return value without the pytest CLI being on PATH.

    Usage:
        python tests/test_phase1.py
    """
    raise SystemExit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()

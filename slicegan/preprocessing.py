"""
SliceGAN preprocessing extended with mat3D support.

Original SliceGAN batch() logic is retained for all existing types.
New type 'mat3D' adds .mat loading + subvolume extraction before patching.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data_pipeline import load_mat_volume, extract_subvolumes, augment_volume

# Total patch cap matching original SliceGAN dataset size
_MAX_PATCHES = 32 * 900  # 28 800


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _onehot_encode(img, phases):
    """One-hot encode a 3D integer label array.

    Parameters
    ----------
    img : numpy.ndarray
        3D integer array with phase labels.
    phases : array-like
        Sorted unique phase labels (e.g. [1, 2, 3]).

    Returns
    -------
    numpy.ndarray
        Float32 array of shape (n_phases, D, H, W).
    """
    n = len(phases)
    out = np.zeros((n, *img.shape), dtype=np.float32)
    for ch, ph in enumerate(phases):
        out[ch] = (img == ph).astype(np.float32)
    return out


def _extract_patches_3d(onehot_vol, l, n_patches):
    """Randomly extract 2D patches from one axis of a one-hot 3D volume.

    Returns patches_x, patches_y, patches_z each as lists of (n_phases, l, l).
    """
    n_phases, D, H, W = onehot_vol.shape
    px, py, pz = [], [], []

    for _ in range(n_patches):
        # x-axis slices: permute (n_phases, D, H, W) → slice along D
        d = np.random.randint(0, D)
        px.append(onehot_vol[:, d, :, :])

        # y-axis slices
        h = np.random.randint(0, H)
        py.append(onehot_vol[:, :, h, :])

        # z-axis slices
        w = np.random.randint(0, W)
        pz.append(onehot_vol[:, :, :, w])

    return px, py, pz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def batch(data, imtype, l, sf):
    """Pre-generate patch datasets for SliceGAN training.

    Parameters
    ----------
    data : list of str
        List of file paths. For isotropic data pass a single-element list;
        it is automatically replicated to three axes.
    imtype : str
        One of: 'mat3D', 'tif3D', 'tif2D', 'png', 'jpg', 'colour',
        'grayscale'.
    l : int
        Patch / slice side length (e.g. 64).
    sf : int
        Scale factor — downsamples via [::sf, ::sf, ::sf] before patching.

    Returns
    -------
    list of TensorDataset
        [dataset_x, dataset_y, dataset_z], each containing one-hot patches
        of shape (n_phases, l, l).
    """
    # Isotropic replication (single path → three identical axes)
    if len(data) == 1:
        data = data * 3

    if imtype == 'mat3D':
        return _batch_mat3d(data, l, sf)
    else:
        return _batch_slicegan_original(data, imtype, l, sf)


# ---------------------------------------------------------------------------
# mat3D implementation
# ---------------------------------------------------------------------------

def _batch_mat3d(data, l, sf):
    """Handle mat3D type: load .mat, extract subvolumes, build patch datasets."""
    # data is already length-3 after isotropic replication;
    # for mat3D we use the first unique path (axes are handled internally).
    seen = []
    paths = []
    for p in data:
        if p not in seen:
            seen.append(p)
            paths.append(p)

    all_px, all_py, all_pz = [], [], []

    for path in paths:
        vol = load_mat_volume(path)

        # Downsample
        if sf > 1:
            vol = vol[::sf, ::sf, ::sf]

        phases = sorted(np.unique(vol).tolist())

        # Extract subvolumes of size l
        subvols = extract_subvolumes(vol, subvol_size=l, stride=l)

        if len(subvols) == 0:
            # Fallback: no coverage filtering if nothing passes
            subvols = extract_subvolumes(vol, subvol_size=l, stride=l,
                                         min_phase_coverage=0.0)

        for sv in subvols:
            # Augment
            sv = augment_volume(sv)
            onehot = _onehot_encode(sv, phases)  # (n_phases, l, l, l)

            # One patch per subvolume per axis (random slice)
            px, py, pz = _extract_patches_3d(onehot, l, n_patches=1)
            all_px.extend(px)
            all_py.extend(py)
            all_pz.extend(pz)

        # Stop early if we have enough
        if len(all_px) >= _MAX_PATCHES:
            break

    # Cap at MAX_PATCHES
    all_px = all_px[:_MAX_PATCHES]
    all_py = all_py[:_MAX_PATCHES]
    all_pz = all_pz[:_MAX_PATCHES]

    def _to_dataset(patches):
        arr = np.stack(patches, axis=0)  # (N, n_phases, l, l)
        return TensorDataset(torch.from_numpy(arr))

    return [_to_dataset(all_px), _to_dataset(all_py), _to_dataset(all_pz)]


# ---------------------------------------------------------------------------
# Original SliceGAN logic (retained for all other types)
# ---------------------------------------------------------------------------

def _batch_slicegan_original(data, imtype, l, sf):
    """Reproduce original SliceGAN batch() behaviour for non-mat types."""
    import tifffile

    datasets = []
    for path in data:
        if imtype in ('tif3D',):
            img = tifffile.imread(path)
            if sf > 1:
                img = img[::sf, ::sf, ::sf]
            phases = np.unique(img)
            onehot = _onehot_encode(img, phases)
            patches = []
            n, D, H, W = onehot.shape
            for _ in range(_MAX_PATCHES):
                axis = np.random.randint(0, 3)
                if axis == 0:
                    idx = np.random.randint(0, D)
                    p = onehot[:, idx, :l, :l]
                elif axis == 1:
                    idx = np.random.randint(0, H)
                    p = onehot[:, :l, idx, :l]
                else:
                    idx = np.random.randint(0, W)
                    p = onehot[:, :l, :l, idx]
                patches.append(p)
            arr = np.stack(patches)
            datasets.append(TensorDataset(torch.from_numpy(arr)))
        else:
            raise NotImplementedError(
                f"imtype '{imtype}' not yet implemented in this extension. "
                "Only 'mat3D' and 'tif3D' are currently supported."
            )

    # If only one dataset was produced (isotropic), replicate
    if len(datasets) == 1:
        datasets = datasets * 3

    return datasets

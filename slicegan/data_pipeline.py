"""
Data pipeline utilities for Conditional SliceGAN.

Handles loading .mat volumes, extracting subvolumes, computing conditioning
vectors (volume fraction + average pore size), and data augmentation.
"""

import numpy as np
import scipy.io
import scipy.ndimage
import torch
from morphology import compute_volume_fraction as compute_vf
from morphology import compute_cld

# ---------------------------------------------------------------------------
# MAT file loading
# ---------------------------------------------------------------------------

def load_mat_volume(path: str) -> np.ndarray:
    """Load a .mat file and return the vol_seg array as a numpy integer array.

    Parameters
    ----------
    path : str or Path
        Path to the .mat file.

    Returns
    -------
    numpy.ndarray
        3D integer array of phase labels (e.g. 1, 2, 3).

    Raises
    ------
    KeyError
        If 'vol_seg' key is not present in the .mat file.
    """
    mat = scipy.io.loadmat(str(path))
    if 'vol_seg' not in mat:
        available = [k for k in mat.keys() if not k.startswith('__')]
        raise KeyError(
            f"'vol_seg' key not found in {path}. "
            f"Available keys: {available}"
        )
    
    vol = mat['vol_seg'].astype(np.int32)
    print(f"[load_mat_volume] shape={vol.shape}, unique labels={np.unique(vol).tolist()}")
    return vol


# ---------------------------------------------------------------------------
# Subvolume extraction
# ---------------------------------------------------------------------------

def extract_subvolumes(volume: np.ndarray, subvol_size: int, stride: int = 0, min_phase_coverage: float = 0.05) -> list:
    """Divide a 3D volume into cubic subvolumes with optional overlap.

    Parameters
    ----------
    volume : numpy.ndarray
        3D integer array of phase labels, shape (X, Y, Z).
    subvol_size : int
        Side length of each cubic subvolume.
    stride : int, optional
        Step between consecutive subvolumes. Defaults to subvol_size (non-overlapping).
    min_phase_coverage : float
        Minimum fraction of voxels that must belong to *each* present phase.
        Subvolumes where any phase occupies less than this fraction are rejected.

    Returns
    -------
    list of numpy.ndarray
        Each element has shape (subvol_size, subvol_size, subvol_size).
    """
    if stride <= 0:
        stride = subvol_size

    X, Y, Z = volume.shape
    s = subvol_size
    phases = np.unique(volume)
    total_voxels = s ** 3 # number of voxels in each subvolume

    subvolumes = []
    n_rejected = 0

    for x in range(0, X - s + 1, stride):
        for y in range(0, Y - s + 1, stride):
            for z in range(0, Z - s + 1, stride):
                patch = volume[x:x+s, y:y+s, z:z+s]
                # Coverage filter: every phase present in the full volume must
                # occupy at least min_phase_coverage of this patch.
                reject = False
                for ph in phases:
                    frac = np.sum(patch == ph) / total_voxels
                    if frac < min_phase_coverage:
                        reject = True
                        break
                if reject:
                    n_rejected += 1
                else:
                    subvolumes.append(patch.copy())

    print(
        f"[extract_subvolumes] extracted={len(subvolumes)}, "
        f"rejected={n_rejected} (min_phase_coverage={min_phase_coverage})"
    )
    return subvolumes

# ---------------------------------------------------------------------------
# Conditioning metrics
# ---------------------------------------------------------------------------

def compute_volume_fraction(volume: torch.Tensor) -> torch.Tensor:
    """Compute per-phase volume fractions from a one-hot encoded volume/slice.

    Formula:
        vf_i = (number of voxels in phase i) / (total number of voxels)

    Parameters
    ----------
    volume : torch.Tensor
        Shape [n_phases, D, H, W] or [n_phases, H, W].

    Returns
    -------
    torch.Tensor
        Shape [n_phases], values sum to 1.0.
    """
    n_phases = volume.shape[0]
    total = volume[0].numel()
    vf = volume.reshape(n_phases, -1).sum(dim=1).float() / total
    return vf


def compute_average_pore_size(binary_mask: np.ndarray) -> float:
    """Compute mean distance-transform value inside a binary phase mask.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary array (2D or 3D). 1 where the phase is present, 0 elsewhere.

    Returns
    -------
    float
        Mean of the Euclidean distance transform within the phase region.
        Returns 0.0 if the phase is absent.
    """
    if binary_mask.sum() == 0:
        return 0.0
    
    dt = scipy.ndimage.distance_transform_edt(binary_mask) # distance to nearest zero (boundary)
    vals = dt[binary_mask > 0]
    return float(vals.mean())


def compute_conditioning_vector(volume: torch.Tensor, n_phases: int) -> torch.Tensor:
    """Build the conditioning vector [vf_0..vf_N, ps_0..ps_N].

    Accepts either:
    - A one-hot encoded volume: numpy array shape (n_phases, *spatial) with
      float values, or
    - A raw integer label volume: numpy array with integer phase labels.

    Parameters
    ----------
    volume : numpy.ndarray
        One-hot encoded (n_phases, ...) or raw integer label volume.
    n_phases : int
        Number of phases.

    Returns
    -------
    numpy.ndarray
        Length 2 * n_phases: [vf_0, ..., vf_{N-1}, ps_0, ..., ps_{N-1}].
    """
    # Detect whether volume is one-hot (first dim == n_phases, float-like)
    # or raw integer labels.
    if volume.ndim > 1 and volume.shape[0] == n_phases and volume.dtype.kind == 'f':
        # One-hot float array: shape (n_phases, *spatial)
        onehot = volume
        labels = np.argmax(onehot, axis=0)  # shape = spatial dims
        unique_phases = list(range(n_phases))
    else:
        # Raw integer label array
        unique_labels = np.unique(volume)
        # Map labels to 0-indexed channels
        label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(unique_labels))}
        labels = np.vectorize(label_to_idx.get)(volume)
        unique_phases = list(range(n_phases))
        onehot = None
    
    phase_labels = dict(zip(unique_phases, range(n_phases)))  
    vf = compute_vf(labels, phase_labels)
    ps, _ = compute_cld(labels, phase_labels, voxel_size_um=0.1, px_min=1.0)
    vf = list(vf.values())
    ps = list(ps.values())
    
    return torch.tensor(vf + ps, dtype=torch.float32)


def compute_dataset_conditioning_stats(dataset_xyz: list, n_phases: int, max_samples: int = 500) -> tuple:
    """Compute mean and std of conditioning vectors across dataset patches.

    Parameters
    ----------
    dataset_xyz : list of TensorDataset
        List of three TensorDatasets [x, y, z], each containing one-hot patches
        of shape [n_phases, 64, 64].
    n_phases : int
        Number of phases.
    max_samples : int
        Maximum number of patches to sample (for speed).

    Returns
    -------
    tuple of (mean, std) numpy arrays, each of length 2 * n_phases.
    """
    vectors = []
    dataset = dataset_xyz[0]  # use x-axis dataset

    indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
    for idx in indices:
        patch = dataset[int(idx)][0].numpy()  # (n_phases, H, W)
        vec = compute_conditioning_vector(patch, n_phases)
        vectors.append(vec)

    vectors = np.stack(vectors, axis=0)
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    return mean, std


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_volume(volume: np.ndarray) -> np.ndarray:
    """Apply random 90° rotations and mirror flips to a 3D volume.

    Parameters
    ----------
    volume : numpy.ndarray
        3D integer array, shape (X, Y, Z).

    Returns
    -------
    numpy.ndarray
        Augmented volume with the same shape.
    """
    # Random 90° rotations on each axis pair
    for axes in [(0, 1), (0, 2), (1, 2)]:
        k = np.random.randint(0, 4)
        if k > 0:
            volume = np.rot90(volume, k=k, axes=axes)

    # Random flips along each axis
    for ax in range(3):
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=ax)

    return np.ascontiguousarray(volume)


"""
Data pipeline utilities for Conditional SliceGAN.

Handles loading .mat volumes, extracting subvolumes, computing conditioning
vectors (volume fraction + average pore size), and data augmentation.
"""

import numpy as np
import scipy.io
import scipy.ndimage

# ---------------------------------------------------------------------------
# MAT file loading
# ---------------------------------------------------------------------------

def load_mat_volume(path: str):
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

def extract_subvolumes(volume: np.ndarray, subvol_size: int, stride: int = 0, min_phase_coverage: float = 0.05):
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



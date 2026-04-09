"""
Data pipeline utilities for Conditional SliceGAN.

Handles loading .mat volumes, extracting subvolumes, computing conditioning
vectors (volume fraction + average pore size), and data augmentation.
"""

import numpy as np
import scipy.io
import scipy.ndimage
import torch


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
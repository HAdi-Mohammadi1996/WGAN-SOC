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

def compute_average_pore_size(volume: np.ndarray, voxel_size: float = 1.0,
    phases: list[int] | None = None, n_lines: int = 500_000,
    px_min_mean: float = 4.0, seed: int | None = 42,) -> np.ndarray:

    """Mean chord length per phase via isotropic ray casting.

    Parameters
    ----------
    volume       : 2-D or 3-D integer-labelled array.
    voxel_size   : physical voxel edge length.
    phases       : labels to evaluate (None = sorted unique labels in volume).
    n_lines      : number of random isotropic rays per phase.
    px_min_mean  : minimum chord length in voxels included in the mean
                   (filters short boundary slivers).
    seed         : RNG seed; per-phase seed = ``seed + phase``.

    Returns
    -------
    np.ndarray of shape (n_phases,) — mean chord length per phase, in the
    same order as `phases` (or sorted unique labels if `phases` is None).
    NaN entries indicate a phase with no chord passing the filter.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")
    if volume.ndim not in (2, 3):
        raise ValueError("volume must be 2-D or 3-D.")

    if phases is None:
        phases = sorted(int(p) for p in np.unique(volume))

    min_len = px_min_mean * voxel_size
    means = np.full(len(phases), np.nan, dtype=np.float64)

    for k, p in enumerate(phases):
        mask = volume == p
        if not mask.any():
            continue

        if volume.ndim == 3:
            chords = _isotropic_chords_3d(
                mask, voxel_size, n_lines,
                None if seed is None else seed + int(p),
            )
        else:
            chords = _isotropic_chords_2d(
                mask, voxel_size, n_lines,
                None if seed is None else seed + int(p),
            )

        kept = chords[chords >= min_len]
        if kept.size:
            means[k] = float(kept.mean())

    return means


def _isotropic_chords_3d(
    mask: np.ndarray,
    voxel_size: float,
    n_lines: int,
    seed: int | None,
) -> np.ndarray:
    """Cast `n_lines` isotropic rays through a 3D mask; return chord lengths in physical units."""
    nx, ny, nz = mask.shape
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, np.array([nx, ny, nz], dtype=float), size=(n_lines, 3))
    dirs = rng.normal(size=(n_lines, 3))
    norms = np.linalg.norm(dirs, axis=1)
    keep = norms > 1e-15
    points = np.ascontiguousarray(points[keep], dtype=np.float64)
    dirs = np.ascontiguousarray(dirs[keep] / norms[keep, None], dtype=np.float64)
    return _trace_3d(mask, points, dirs, px_min=1.0, voxel_size=voxel_size)


def _isotropic_chords_2d(
    mask: np.ndarray,
    voxel_size: float,
    n_lines: int,
    seed: int | None,
) -> np.ndarray:
    """Cast `n_lines` isotropic rays through a 2D mask; return chord lengths in physical units."""
    nx, ny = mask.shape
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, np.array([nx, ny], dtype=float), size=(n_lines, 2))
    dirs = rng.normal(size=(n_lines, 2))
    norms = np.linalg.norm(dirs, axis=1)
    keep = norms > 1e-15
    points = np.ascontiguousarray(points[keep], dtype=np.float64)
    dirs = np.ascontiguousarray(dirs[keep] / norms[keep, None], dtype=np.float64)
    return _trace_2d(mask, points, dirs, px_min=1.0, voxel_size=voxel_size)


def _trace_3d(vol, points, dirs, px_min, voxel_size):
    """Vectorised Amanatides-Woo voxel traversal through a 3D volume."""
    N = points.shape[0]
    nx, ny, nz = vol.shape
    eps = 1e-12
    tol = 1e-12
    INF = 1e300
    tiny = 1e-15

    dx = dirs[:, 0].copy()
    dy = dirs[:, 1].copy()
    dz = dirs[:, 2].copy()
    p0 = points[:, 0].copy()
    p1 = points[:, 1].copy()
    p2 = points[:, 2].copy()

    def slab(p, d, L):
        """Parametric entry/exit interval [lo, hi] for 0 <= p + t*d <= L."""
        safe_d = np.where(np.abs(d) > tiny, d, 1.0)
        t1 = -p / safe_d
        t2 = (L - p) / safe_d
        lo = np.minimum(t1, t2)
        hi = np.maximum(t1, t2)
        parallel = np.abs(d) <= tiny
        outside = parallel & ((p < 0.0) | (p > L))
        inside = parallel & ~outside
        lo = np.where(inside, -INF, lo)
        hi = np.where(inside, INF, hi)
        lo = np.where(outside, INF, lo)
        hi = np.where(outside, -INF, hi)
        return lo, hi

    lox, hix = slab(p0, dx, float(nx))
    loy, hiy = slab(p1, dy, float(ny))
    loz, hiz = slab(p2, dz, float(nz))
    t_lo = np.maximum(np.maximum(lox, loy), loz)
    t_hi = np.minimum(np.minimum(hix, hiy), hiz)

    active = t_hi > t_lo
    if not active.any():
        return np.empty(0, dtype=np.float64)

    # Entry voxel: nudge by eps so we land just inside the first voxel
    t = np.where(active, t_lo, 0.0)
    ex = p0 + (t + eps) * dx
    ey = p1 + (t + eps) * dy
    ez = p2 + (t + eps) * dz
    ix = np.clip(np.floor(ex).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor(ey).astype(np.int64), 0, ny - 1)
    iz = np.clip(np.floor(ez).astype(np.int64), 0, nz - 1)

    def aw_setup(d, p, i):
        """Amanatides-Woo per-axis (step, t_max, t_delta)."""
        safe_d = np.where(np.abs(d) > tiny, d, 1.0)
        t_max = np.where(
            d > 0, (i + 1.0 - p) / safe_d,
            np.where(d < 0, (i - p) / safe_d, INF),
        )
        t_delta = np.where(
            d > 0, 1.0 / safe_d,
            np.where(d < 0, -1.0 / safe_d, INF),
        )
        step = np.where(d > 0, 1, np.where(d < 0, -1, 0)).astype(np.int64)
        return step, t_max, t_delta

    step_x, t_max_x, t_d_x = aw_setup(dx, p0, ix)
    step_y, t_max_y, t_d_y = aw_setup(dy, p1, iy)
    step_z, t_max_z, t_d_z = aw_setup(dz, p2, iz)

    # Physical length per unit t along the ray (|d * voxel_size|; = voxel_size for unit d, isotropic voxels)
    scale = np.sqrt((dx * voxel_size) ** 2 + (dy * voxel_size) ** 2 + (dz * voxel_size) ** 2)

    cur = vol[ix, iy, iz].astype(np.int64)
    run = np.zeros(N, dtype=np.float64)
    chord_buf: list[np.ndarray] = []

    max_iters = 3 * (nx + ny + nz) + 10
    for _ in range(max_iters):
        if not active.any():
            break

        # Distance to next voxel boundary or box exit
        t_next = np.minimum(np.minimum(t_max_x, t_max_y), t_max_z)
        t_next = np.minimum(t_next, t_hi)
        seg = t_next - t

        # Phase at current voxel
        ix_c = np.clip(ix, 0, nx - 1)
        iy_c = np.clip(iy, 0, ny - 1)
        iz_c = np.clip(iz, 0, nz - 1)
        ph = vol[ix_c, iy_c, iz_c].astype(np.int64)

        same = (ph == cur) & active
        change = (~same) & active
        # Extend the current-phase run on same-phase segments
        run[same] += seg[same]
        # Emit chord on phase change (only if previous phase was the target and run is long enough)
        emit = change & (run > px_min) & (cur != 0)
        if emit.any():
            chord_buf.append(run[emit] * scale[emit])
        # Reset run + update tracked phase on change
        run[change] = seg[change]
        cur[change] = ph[change]

        # Advance voxel coords along the axis (or axes, if tied) that hit t_next first
        adv_x = (np.abs(t_max_x - t_next) <= tol) & active
        adv_y = (np.abs(t_max_y - t_next) <= tol) & active
        adv_z = (np.abs(t_max_z - t_next) <= tol) & active
        ix[adv_x] += step_x[adv_x]
        iy[adv_y] += step_y[adv_y]
        iz[adv_z] += step_z[adv_z]
        t_max_x[adv_x] += t_d_x[adv_x]
        t_max_y[adv_y] += t_d_y[adv_y]
        t_max_z[adv_z] += t_d_z[adv_z]

        t = t_next
        # Deactivate rays that exited the box
        active &= (
            (t < t_hi - eps)
            & (ix >= 0) & (ix < nx)
            & (iy >= 0) & (iy < ny)
            & (iz >= 0) & (iz < nz)
        )

    # Flush any run still open at ray exit
    final_emit = (run > px_min) & (cur != 0)
    if final_emit.any():
        chord_buf.append(run[final_emit] * scale[final_emit])

    if not chord_buf:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chord_buf)


def _trace_2d(vol, points, dirs, px_min, voxel_size):
    """Vectorised Amanatides-Woo voxel traversal through a 2D image."""
    N = points.shape[0]
    nx, ny = vol.shape
    eps = 1e-12
    tol = 1e-12
    INF = 1e300
    tiny = 1e-15

    dx = dirs[:, 0].copy()
    dy = dirs[:, 1].copy()
    p0 = points[:, 0].copy()
    p1 = points[:, 1].copy()

    def slab(p, d, L):
        safe_d = np.where(np.abs(d) > tiny, d, 1.0)
        t1 = -p / safe_d
        t2 = (L - p) / safe_d
        lo = np.minimum(t1, t2)
        hi = np.maximum(t1, t2)
        parallel = np.abs(d) <= tiny
        outside = parallel & ((p < 0.0) | (p > L))
        inside = parallel & ~outside
        lo = np.where(inside, -INF, lo)
        hi = np.where(inside, INF, hi)
        lo = np.where(outside, INF, lo)
        hi = np.where(outside, -INF, hi)
        return lo, hi

    lox, hix = slab(p0, dx, float(nx))
    loy, hiy = slab(p1, dy, float(ny))
    t_lo = np.maximum(lox, loy)
    t_hi = np.minimum(hix, hiy)

    active = t_hi > t_lo
    if not active.any():
        return np.empty(0, dtype=np.float64)

    t = np.where(active, t_lo, 0.0)
    ex = p0 + (t + eps) * dx
    ey = p1 + (t + eps) * dy
    ix = np.clip(np.floor(ex).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor(ey).astype(np.int64), 0, ny - 1)

    def aw_setup(d, p, i):
        safe_d = np.where(np.abs(d) > tiny, d, 1.0)
        t_max = np.where(
            d > 0, (i + 1.0 - p) / safe_d,
            np.where(d < 0, (i - p) / safe_d, INF),
        )
        t_delta = np.where(
            d > 0, 1.0 / safe_d,
            np.where(d < 0, -1.0 / safe_d, INF),
        )
        step = np.where(d > 0, 1, np.where(d < 0, -1, 0)).astype(np.int64)
        return step, t_max, t_delta

    step_x, t_max_x, t_d_x = aw_setup(dx, p0, ix)
    step_y, t_max_y, t_d_y = aw_setup(dy, p1, iy)

    scale = np.sqrt((dx * voxel_size) ** 2 + (dy * voxel_size) ** 2)

    cur = vol[ix, iy].astype(np.int64)
    run = np.zeros(N, dtype=np.float64)
    chord_buf: list[np.ndarray] = []

    max_iters = 2 * (nx + ny) + 10
    for _ in range(max_iters):
        if not active.any():
            break

        t_next = np.minimum(t_max_x, t_max_y)
        t_next = np.minimum(t_next, t_hi)
        seg = t_next - t

        ix_c = np.clip(ix, 0, nx - 1)
        iy_c = np.clip(iy, 0, ny - 1)
        ph = vol[ix_c, iy_c].astype(np.int64)

        same = (ph == cur) & active
        change = (~same) & active
        run[same] += seg[same]
        emit = change & (run > px_min) & (cur != 0)
        if emit.any():
            chord_buf.append(run[emit] * scale[emit])
        run[change] = seg[change]
        cur[change] = ph[change]

        adv_x = (np.abs(t_max_x - t_next) <= tol) & active
        adv_y = (np.abs(t_max_y - t_next) <= tol) & active
        ix[adv_x] += step_x[adv_x]
        iy[adv_y] += step_y[adv_y]
        t_max_x[adv_x] += t_d_x[adv_x]
        t_max_y[adv_y] += t_d_y[adv_y]

        t = t_next
        active &= (
            (t < t_hi - eps)
            & (ix >= 0) & (ix < nx)
            & (iy >= 0) & (iy < ny)
        )

    final_emit = (run > px_min) & (cur != 0)
    if final_emit.any():
        chord_buf.append(run[final_emit] * scale[final_emit])

    if not chord_buf:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chord_buf)


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

    if onehot is not None:
        vf = compute_volume_fraction(torch.from_numpy(onehot).float())
    else:
        vf = torch.tensor(
            [(labels == i).mean() for i in range(n_phases)],
            dtype=torch.float32
        )
    
    # vf = compute_volume_fraction(torch.from_numpy(volume).float())
    ps = compute_average_pore_size(labels, voxel_size=0.1, px_min_mean=4.0)
    ps = torch.from_numpy(ps).float()
    vf = list(vf.numpy())
    ps = list(ps.numpy())

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


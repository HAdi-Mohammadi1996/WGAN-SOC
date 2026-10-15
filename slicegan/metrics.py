"""
Microstructure metrics and validation utilities for Conditional SliceGAN.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import ks_2samp

from slicegan.data_pipeline import compute_average_pore_size

# Fixed phase colours: label k is always drawn in _PHASE_COLOURS[k].
_PHASE_COLOURS = ['#000000', '#7f7f7f', '#ffffff', '#1f77b4', '#ff7f0e',
                  '#2ca02c', '#d62728', '#9467bd']


class MicrostructureMetrics:
    """Scalar microstructure metrics for integer-labelled 3-D volumes."""

    def volume_fraction(self, volume: np.ndarray, n_phases: int) -> np.ndarray:
        """Per-phase volume fractions summing to 1.0.

        Parameters
        ----------
        volume : np.ndarray
            Integer-labelled 3-D array with values in [0, n_phases-1].
        n_phases : int
            Total number of phases.

        Returns
        -------
        np.ndarray
            Shape (n_phases,), values sum to 1.0.
        """
        total = volume.size
        return np.array(
            [(volume == p).sum() / total for p in range(n_phases)],
            dtype=np.float64,
        )

    def average_pore_size(
        self, volume: np.ndarray, phase_idx: int, n_lines: int = 500_000
    ) -> float:
        """Mean chord length for one phase via isotropic chord sampling.

        Parameters
        ----------
        volume : np.ndarray
            Integer-labelled 3-D array.
        phase_idx : int
            Phase label to measure.
        n_lines : int
            Number of isotropic rays cast (lower is faster but noisier).

        Returns
        -------
        float
            Mean chord length in voxels; 0.0 if phase is absent.
        """
        ps = compute_average_pore_size(volume, phases=[phase_idx], n_lines=n_lines)
        return float(np.nan_to_num(ps, nan=0.0)[0])

    def relative_surface_area(self, volume: np.ndarray, phase_idx: int) -> float:
        """Fraction of voxel faces shared between phase_idx and any other phase.

        Parameters
        ----------
        volume : np.ndarray
            Integer-labelled 3-D array.
        phase_idx : int
            Phase label for which to measure surface area.

        Returns
        -------
        float
            Interface face count divided by total voxel count.
        """
        f = (volume == phase_idx).astype(np.int8)
        sa = sum(int(np.abs(np.diff(f, axis=a)).sum()) for a in range(f.ndim))
        return float(sa / volume.size)

    def two_point_correlation(
        self, volume: np.ndarray, phase_idx: int, max_r: int
    ) -> np.ndarray:
        """S₂(r) for integer lags 0..max_r along axis 0.

        S₂(r) = P(voxel x is in phase AND voxel x+r is in phase).
        At r=0 equals volume fraction; approaches vf² at large r (independence).
        Uses periodic shift (np.roll), so wrap-around affects large r on small volumes.

        Parameters
        ----------
        volume : np.ndarray
            Integer-labelled 3-D array.
        phase_idx : int
            Phase label.
        max_r : int
            Maximum lag to compute.

        Returns
        -------
        np.ndarray
            Length max_r+1; index r gives S₂(r).
        """
        f = (volume == phase_idx).astype(np.float64)
        return np.array(
            [(f * np.roll(f, r, axis=0)).mean() for r in range(max_r + 1)]
        )

    def triple_phase_boundary_density(self, volume: np.ndarray) -> float:
        """Fraction of 2×2 face-plane quads containing all 3 phases.

        Scans all three families of axis-aligned planes; for each 2×2 quad
        checks whether phases 0, 1, and 2 are all represented. Intended for
        3-phase volumes (labels 0, 1, 2).

        Parameters
        ----------
        volume : np.ndarray
            Integer-labelled 3-D array with values in {0, 1, 2}.

        Returns
        -------
        float
            TPB quad count divided by volume.size (density per voxel).
        """
        if volume.max() > 2 or volume.min() < 0:
            raise ValueError(
                "triple_phase_boundary_density requires a 3-phase volume with "
                f"labels in {{0, 1, 2}}; got range [{volume.min()}, {volume.max()}]."
            )
        v = volume
        tpb = 0

        # XY-plane quads: edges parallel to Z axis
        q = np.stack(
            [v[:, :-1, :-1], v[:, 1:, :-1], v[:, :-1, 1:], v[:, 1:, 1:]], axis=-1
        )
        tpb += int(
            (np.any(q == 0, axis=-1) & np.any(q == 1, axis=-1) & np.any(q == 2, axis=-1)).sum()
        )

        # XZ-plane quads: edges parallel to Y axis
        q = np.stack(
            [v[:-1, :, :-1], v[1:, :, :-1], v[:-1, :, 1:], v[1:, :, 1:]], axis=-1
        )
        tpb += int(
            (np.any(q == 0, axis=-1) & np.any(q == 1, axis=-1) & np.any(q == 2, axis=-1)).sum()
        )

        # YZ-plane quads: edges parallel to X axis
        q = np.stack(
            [v[:-1, :-1, :], v[1:, :-1, :], v[:-1, 1:, :], v[1:, 1:, :]], axis=-1
        )
        tpb += int(
            (np.any(q == 0, axis=-1) & np.any(q == 1, axis=-1) & np.any(q == 2, axis=-1)).sum()
        )

        return tpb / volume.size

    def inter_slice_coherence(self, volume: np.ndarray) -> float:
        """Mean fraction of voxels with the same phase in adjacent slice pairs.

        Computed over all 3 axes: for each axis, average the per-pair agreement
        across all consecutive slice pairs, then average over axes.

        Parameters
        ----------
        volume : np.ndarray
            Integer-labelled 3-D array.

        Returns
        -------
        float
            Value in [0, 1]; 1.0 for a perfectly uniform volume.
        """
        total = 0.0
        count = 0
        for a in range(volume.ndim):
            v = np.moveaxis(volume, a, 0)
            for i in range(v.shape[0] - 1):
                total += float((v[i] == v[i + 1]).mean())
                count += 1
        return total / count if count > 0 else 1.0


# ---------------------------------------------------------------------------
# Standalone validation functions
# ---------------------------------------------------------------------------

def _r2_mae(targets: np.ndarray, predicted: np.ndarray) -> tuple:
    """R² and MAE over finite entries; (nan, nan) if none are finite."""
    ok = np.isfinite(targets) & np.isfinite(predicted)
    if not ok.any():
        return float('nan'), float('nan')
    t, p = targets[ok], predicted[ok]
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(r2), float(np.abs(t - p).mean())


def _scatter_plot(targets, predicted, n_phases, label, path):
    """Target vs. generated scatter, one colour per phase, with a y=x line."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for p in range(n_phases):
        ax.scatter(targets[:, p], predicted[:, p], color=_PHASE_COLOURS[p % len(_PHASE_COLOURS)],
                   label=f'phase {p}', s=25)
    finite = np.concatenate([targets[np.isfinite(targets)], predicted[np.isfinite(predicted)]])
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel(f'target {label}')
    ax.set_ylabel(f'generated {label}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def conditioning_accuracy_test(
    netG,
    cond_enc,
    test_vectors,
    n_samples: int,
    nz: int,
    lz: int,
    norm_stats,
    device,
    save_dir: str | None = None,
    ps_n_lines: int = 20_000,
) -> dict:
    """Measure how accurately the generator reproduces target conditioning.

    For each test vector generates n_samples volumes. VF is read from the
    soft (softmax) output; pore size is measured on the argmax-segmented
    volumes and averaged over samples. R² and MAE are computed against the
    targets in the first (VF) and second (PS) halves of each vector.

    Parameters
    ----------
    netG : Generator3D
        Generator (may be untrained for a smoke-test).
    cond_enc : ConditioningEncoder
        Conditioning encoder.
    test_vectors : list of array-like
        Raw (unnormalised) conditioning vectors, each of length 2*n_phases.
    n_samples : int
        Volumes to generate per test vector.
    nz : int
        Latent noise channels.
    lz : int
        Latent spatial size.
    norm_stats : tuple of (mean_np, std_np)
        Normalisation stats from training.
    device : torch.device or str
    save_dir : str, optional
        If given, writes ``vf_scatter.png`` and ``ps_scatter.png`` here.
    ps_n_lines : int
        Rays per phase for the chord-length pore-size estimate.

    Returns
    -------
    dict
        ``r2``/``mae`` (VF), ``r2_ps``/``mae_ps`` (pore size; NaN entries
        from absent phases are excluded), and the per-vector arrays
        ``vf_pred``, ``vf_target``, ``ps_pred``, ``ps_target`` of shape
        ``[n_vectors, n_phases]``.
    """
    from slicegan.model import generate_conditioned_samples

    m = MicrostructureMetrics()
    n_phases = len(test_vectors[0]) // 2
    vf_pred, vf_target, ps_pred, ps_target = [], [], [], []

    for vec in test_vectors:
        samples = generate_conditioned_samples(
            netG, cond_enc, vec, n_samples, nz, lz, norm_stats, device
        )
        # samples: [n_samples, n_phases, l, l, l] soft output
        vf_pred.append(samples.mean(dim=[0, 2, 3, 4]).detach().cpu().numpy())
        vf_target.append(np.asarray(vec[:n_phases], dtype=np.float64))

        hard = samples.argmax(dim=1).cpu().numpy()  # [n_samples, l, l, l]
        ps = np.array([
            [m.average_pore_size(v, p, n_lines=ps_n_lines) for p in range(n_phases)]
            for v in hard
        ])
        ps[ps == 0.0] = np.nan  # absent phase / no chords passed the filter
        with np.errstate(all='ignore'):
            ps_mean = np.nanmean(ps, axis=0) if np.isfinite(ps).any() else np.full(n_phases, np.nan)
        ps_pred.append(ps_mean)
        ps_target.append(np.asarray(vec[n_phases:2 * n_phases], dtype=np.float64))

    vf_pred = np.array(vf_pred, dtype=np.float64)
    vf_target = np.array(vf_target)
    ps_pred = np.array(ps_pred, dtype=np.float64)
    ps_target = np.array(ps_target)

    r2, mae = _r2_mae(vf_target.ravel(), vf_pred.ravel())
    r2_ps, mae_ps = _r2_mae(ps_target.ravel(), ps_pred.ravel())

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _scatter_plot(vf_target, vf_pred, n_phases, 'VF',
                      os.path.join(save_dir, 'vf_scatter.png'))
        _scatter_plot(ps_target, ps_pred, n_phases, 'pore size',
                      os.path.join(save_dir, 'ps_scatter.png'))

    return {
        'r2': r2, 'mae': mae, 'r2_ps': r2_ps, 'mae_ps': mae_ps,
        'vf_pred': vf_pred, 'vf_target': vf_target,
        'ps_pred': ps_pred, 'ps_target': ps_target,
    }


def _metric_table(volumes: list, n_phases: int, ps_n_lines: int) -> dict:
    """Per-volume scalar metrics, keyed by metric name → array over volumes."""
    m = MicrostructureMetrics()
    table = {}
    vfs = np.array([m.volume_fraction(v, n_phases) for v in volumes])
    for p in range(n_phases):
        table[f'vf_phase{p}'] = vfs[:, p]
    for p in range(n_phases):
        table[f'ps_phase{p}'] = np.array(
            [m.average_pore_size(v, p, n_lines=ps_n_lines) for v in volumes]
        )
    for p in range(n_phases):
        table[f'sa_phase{p}'] = np.array(
            [m.relative_surface_area(v, p) for v in volumes]
        )
    table['coherence'] = np.array([m.inter_slice_coherence(v) for v in volumes])
    if n_phases == 3:
        table['tpb'] = np.array([m.triple_phase_boundary_density(v) for v in volumes])
    return table


def statistical_similarity_test(
    real_volumes: list,
    generated_volumes: list,
    save_path: str | None = None,
    ps_n_lines: int = 20_000,
) -> dict:
    """Two-sample KS test per microstructure metric.

    Metrics per volume: volume fraction, pore size and relative surface
    area per phase; inter-slice coherence; TPB density (3-phase only).

    Parameters
    ----------
    real_volumes : list of np.ndarray
        Integer-labelled 3-D real volumes.
    generated_volumes : list of np.ndarray
        Integer-labelled 3-D generated volumes.
    save_path : str, optional
        If given, saves real-vs-generated box plots (SliceGAN Fig. 4 style),
        one panel per metric, to this path.
    ps_n_lines : int
        Rays per phase for the chord-length pore-size estimate.

    Returns
    -------
    dict
        Mapping metric name (``vf_phase{p}``, ``ps_phase{p}``,
        ``sa_phase{p}``, ``coherence``, ``tpb``) → ``{'statistic', 'pvalue'}``.
    """
    n_phases = max(
        int(max(v.max() for v in real_volumes)),
        int(max(v.max() for v in generated_volumes)),
    ) + 1

    real_t = _metric_table(real_volumes, n_phases, ps_n_lines)
    gen_t = _metric_table(generated_volumes, n_phases, ps_n_lines)

    result = {}
    for key in real_t:
        stat, pval = ks_2samp(real_t[key], gen_t[key])
        result[key] = {'statistic': float(stat), 'pvalue': float(pval)}

    if save_path is not None:
        _box_plots(real_t, gen_t, save_path)
    return result


def _box_plots(real_t: dict, gen_t: dict, save_path: str) -> None:
    """Grid of real-vs-generated box plots, one panel per metric."""
    keys = list(real_t)
    ncols = min(len(keys), 4)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        bp = ax.boxplot([real_t[key], gen_t[key]], patch_artist=True, widths=0.6)
        ax.set_xticks([1, 2], ['real', 'gen'])
        for patch, colour in zip(bp['boxes'], ['#1f77b4', '#ff7f0e']):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        ax.set_title(key, fontsize=9)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def edge_centre_consistency(
    volume: np.ndarray, shell: int = 4, tol: float = 0.2, n_phases: int | None = None
) -> dict:
    """Compare per-phase VF in the outer shell against the central core.

    A large difference indicates the uniform-information-density rules were
    violated (edge artefacts in the generator output).

    Parameters
    ----------
    volume : np.ndarray
        Integer-labelled 3-D array.
    shell : int
        Shell thickness in voxels.
    tol : float
        Maximum allowed relative VF difference, per phase.
    n_phases : int, optional
        Defaults to ``volume.max() + 1``.

    Returns
    -------
    dict
        ``edge`` and ``centre`` VF arrays, ``rel_diff`` per phase
        (|edge - centre| / centre; 0 where both are 0), and ``passed``.
    """
    if min(volume.shape) <= 2 * shell:
        raise ValueError(f"volume {volume.shape} too small for shell={shell}")
    if n_phases is None:
        n_phases = int(volume.max()) + 1
    core = np.zeros(volume.shape, dtype=bool)
    core[shell:-shell, shell:-shell, shell:-shell] = True
    edge_vals, centre_vals = volume[~core], volume[core]
    edge = np.array([(edge_vals == p).mean() for p in range(n_phases)])
    centre = np.array([(centre_vals == p).mean() for p in range(n_phases)])
    diff = np.abs(edge - centre)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(centre > 0, diff / np.where(centre > 0, centre, 1.0),
                       np.where(diff > 0, np.inf, 0.0))
    return {'edge': edge, 'centre': centre, 'rel_diff': rel,
            'passed': bool((rel <= tol).all())}


def phase_colormap(n_phases: int) -> tuple:
    """Fixed (cmap, norm) so label k always maps to ``_PHASE_COLOURS[k]``."""
    colours = [_PHASE_COLOURS[k % len(_PHASE_COLOURS)] for k in range(n_phases)]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(n_phases + 1) - 0.5, n_phases)
    return cmap, norm


def visualise_volume(
    volume, imtype: str, save_path: str, n_phases: int | None = None
) -> None:
    """Save three orthogonal mid-plane cross-sections of a volume to a PNG.

    Parameters
    ----------
    volume : np.ndarray or torch.Tensor
        Integer-labelled ``[D, H, W]`` array, or one-hot / softmax
        ``[nc, D, H, W]`` (argmaxed over the channel axis).
    imtype : str
        Image type tag (e.g. 'nphase'); kept for SliceGAN API parity.
    save_path : str
        Full output path including file extension (e.g. 'result.png').
    n_phases : int, optional
        Number of phases for the colour map. Defaults to the channel count
        for one-hot input, else ``max(volume.max() + 1, 3)``. Label k is
        always drawn in the same colour regardless of which labels occur.
    """
    if hasattr(volume, 'detach'):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume)
    if volume.ndim == 4:
        if n_phases is None:
            n_phases = volume.shape[0]
        volume = volume.argmax(axis=0)
    if n_phases is None:
        n_phases = max(int(volume.max()) + 1, 3)

    D, H, W = volume.shape
    cmap, norm = phase_colormap(n_phases)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices = [
        volume[D // 2, :, :],
        volume[:, H // 2, :],
        volume[:, :, W // 2],
    ]
    titles = ['XY (mid-Z)', 'XZ (mid-Y)', 'YZ (mid-X)']

    for ax, sl, title in zip(axes, slices, titles):
        ax.imshow(sl, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

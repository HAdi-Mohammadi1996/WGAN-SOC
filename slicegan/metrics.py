"""
Microstructure metrics and validation utilities for Conditional SliceGAN.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from slicegan.data_pipeline import compute_average_pore_size


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

    def average_pore_size(self, volume: np.ndarray, phase_idx: int) -> float:
        """Mean chord length for one phase via isotropic chord sampling.

        Parameters
        ----------
        volume : np.ndarray
            Integer-labelled 3-D array.
        phase_idx : int
            Phase label to measure.

        Returns
        -------
        float
            Mean chord length in voxels; 0.0 if phase is absent.
        """
        ps = compute_average_pore_size(volume, phases=[phase_idx])
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

def conditioning_accuracy_test(
    netG,
    cond_enc,
    test_vectors,
    n_samples: int,
    nz: int,
    lz: int,
    norm_stats,
    device,
) -> dict:
    """Measure how accurately the generator reproduces target conditioning.

    For each test vector generates n_samples volumes, extracts the mean VF
    from the soft (softmax) output, and computes R² and MAE against the
    target VF values encoded in the first n_phases entries of each vector.

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

    Returns
    -------
    dict
        {'r2': float, 'mae': float}
    """
    from slicegan.model import generate_conditioned_samples

    n_phases = len(test_vectors[0]) // 2
    predicted = []
    targets = []

    for vec in test_vectors:
        samples = generate_conditioned_samples(
            netG, cond_enc, vec, n_samples, nz, lz, norm_stats, device
        )
        # samples: [n_samples, n_phases, l, l, l] soft output
        actual_vf = samples.mean(dim=[0, 2, 3, 4]).detach().cpu().numpy()
        target_vf = np.array(vec[:n_phases], dtype=np.float64)
        predicted.append(actual_vf)
        targets.append(target_vf)

    predicted = np.array(predicted).ravel()
    targets = np.array(targets).ravel()

    ss_res = float(((targets - predicted) ** 2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    mae = float(np.abs(targets - predicted).mean())
    return {'r2': float(r2), 'mae': mae}


def statistical_similarity_test(
    real_volumes: list,
    generated_volumes: list,
) -> dict:
    """KS test per phase on volume fraction distributions.

    Parameters
    ----------
    real_volumes : list of np.ndarray
        Integer-labelled 3-D real volumes.
    generated_volumes : list of np.ndarray
        Integer-labelled 3-D generated volumes.

    Returns
    -------
    dict
        Mapping 'vf_phase{p}' → {'statistic': float, 'pvalue': float}
        for each phase p.
    """
    m = MicrostructureMetrics()
    n_phases = max(
        int(max(v.max() for v in real_volumes)),
        int(max(v.max() for v in generated_volumes)),
    ) + 1

    real_vfs = np.array([m.volume_fraction(v, n_phases) for v in real_volumes])
    gen_vfs = np.array([m.volume_fraction(v, n_phases) for v in generated_volumes])

    result = {}
    for p in range(n_phases):
        stat, pval = ks_2samp(real_vfs[:, p], gen_vfs[:, p])
        result[f'vf_phase{p}'] = {'statistic': float(stat), 'pvalue': float(pval)}
    return result


def visualise_volume(volume: np.ndarray, imtype: str, save_path: str) -> None:
    """Save three orthogonal mid-plane cross-sections of a volume to a PNG.

    Parameters
    ----------
    volume : np.ndarray
        Integer-labelled 3-D array of shape [D, H, W].
    imtype : str
        Image type tag (e.g. 'nphase'); reserved for future colour schemes.
    save_path : str
        Full output path including file extension (e.g. 'result.png').
    """
    D, H, W = volume.shape
    n_phases = int(volume.max()) + 1
    cmap = matplotlib.colormaps['tab10']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices = [
        volume[D // 2, :, :],
        volume[:, H // 2, :],
        volume[:, :, W // 2],
    ]
    titles = ['XY (mid-Z)', 'XZ (mid-Y)', 'YZ (mid-X)']

    for ax, sl, title in zip(axes, slices, titles):
        ax.imshow(sl, cmap=cmap, vmin=0, vmax=max(n_phases - 1, 1),
                  interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

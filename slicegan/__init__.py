"""
slicegan — Conditional SliceGAN package.

"""

from .preprocessing import batch

from .data_pipeline import (
    load_mat_volume,
    extract_subvolumes,
    compute_volume_fraction,
    compute_average_pore_size,
    compute_conditioning_vector,
    compute_dataset_conditioning_stats,
    augment_volume,
)

__all__ = [
    # Preprocessing
    "batch",
    # Data loading
    "load_mat_volume",
    "extract_subvolumes",
    # Conditioning metrics
    "compute_volume_fraction",
    "compute_average_pore_size",
    "compute_conditioning_vector",
    "compute_dataset_conditioning_stats",
    # Augmentation
    "augment_volume",
]

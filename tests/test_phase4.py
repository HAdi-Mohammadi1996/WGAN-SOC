import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch

from slicegan import data_pipeline
from slicegan import model


def test_sample_conditioning_from_batch_normalises(monkeypatch):
    def fake_conditioning_vector(patch, n_phases):
        assert patch.shape[0] == n_phases
        return torch.tensor([
            patch[0].mean(),
            patch[1].mean(),
            patch[0].sum(),
            patch[1].sum(),
        ])

    monkeypatch.setattr(model, "compute_conditioning_vector", fake_conditioning_vector)

    patches = torch.tensor([
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
    ])
    mean = torch.tensor([0.5, 0.5, 2.0, 2.0])
    std = torch.tensor([0.5, 0.5, 2.0, 2.0])

    cond = model.sample_conditioning_from_batch(patches, (mean, std))

    raw = torch.tensor([
        [1.0, 0.0, 4.0, 0.0],
        [0.25, 0.75, 1.0, 3.0],
    ])
    expected = (raw - mean) / std
    assert torch.allclose(cond, expected)


def test_sample_conditioning_from_batch_requires_batched_patches():
    with pytest.raises(ValueError):
        model.sample_conditioning_from_batch(torch.zeros(3, 8, 8), (torch.zeros(6), torch.ones(6)))


def test_conditioning_vector_keeps_absent_phase_slots(monkeypatch):
    def fake_pore_size(labels, voxel_size, phases, px_min_mean):
        assert phases == [0, 1, 2]
        return [1.0, float("nan"), float("nan")]

    monkeypatch.setattr(data_pipeline, "compute_average_pore_size", fake_pore_size)

    patch = torch.zeros(3, 4, 4).numpy()
    patch[0] = 1.0

    cond = data_pipeline.compute_conditioning_vector(patch, n_phases=3)

    assert cond.shape == (6,)
    assert torch.allclose(cond, torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

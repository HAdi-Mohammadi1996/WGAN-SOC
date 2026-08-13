# Phase 5: Validation and Evaluation

**Goal**: Quantify the quality of generated microstructures relative to real data and verify that
conditioning vectors are reflected in the outputs.

**New files**:
- `slicegan/metrics.py` — `MicrostructureMetrics` class + standalone validation functions
- `tests/test_phase5.py` — 10 tests

**`MicrostructureMetrics` class** (all methods accept integer-labelled 3-D `np.ndarray` unless noted):

| Method | Signature | Returns |
|--------|-----------|---------|
| `volume_fraction` | `(volume, n_phases)` | `np.ndarray` shape `(n_phases,)`, sums to 1.0 |
| `average_pore_size` | `(volume, phase_idx)` | `float` — mean chord length (voxels) |
| `relative_surface_area` | `(volume, phase_idx)` | `float` — interface faces / total voxels |
| `two_point_correlation` | `(volume, phase_idx, max_r)` | `np.ndarray` length `max_r+1`, S₂(r) |
| `triple_phase_boundary_density` | `(volume)` | `float` — TPB edges / total voxels (3-phase only) |
| `inter_slice_coherence` | `(volume)` | `float` in [0, 1] — mean adjacent-slice agreement |

**Standalone functions**:

| Function | Returns |
|----------|---------|
| `conditioning_accuracy_test(netG, cond_enc, test_vectors, n_samples, nz, lz, norm_stats, device)` | `dict` with `'r2'` and `'mae'` floats |
| `statistical_similarity_test(real_volumes, generated_volumes)` | `dict` mapping metric name → `{'statistic', 'pvalue'}` |
| `visualise_volume(volume, imtype, save_path)` | None — saves PNG at `save_path` |

**Two-point correlation** S₂(r): probability that two voxels separated by lag r (along axis 0)
are both in `phase_idx`. At r=0 equals VF; approaches VF² for large r (independence).

**Inter-slice coherence**: for each axis, fraction of voxels with same phase in adjacent slice
pairs, averaged over all pairs and all 3 axes.

**TPB density**: fraction of face-edges (in each of 3 axis-aligned planes) whose surrounding
2×2 quad contains all 3 phases. Defined only for 3-phase volumes.

**Tests — `tests/test_phase5.py`** (all must pass before proceeding):

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_vf_matches_phase1` | Metrics VF equals `compute_volume_fraction` from data_pipeline | Identical to 1e-6 |
| `test_s2_at_zero_equals_vf` | S₂(0) == volume fraction | Within 1e-4 |
| `test_s2_at_infinity` | S₂(max_r) ≈ vf² | Within 5 % of `vf²` |
| `test_ks_same_distribution` | KS accepts matched VF samples | p > 0.05 |
| `test_ks_different_distributions` | KS rejects mismatched VF samples | p < 0.05 |
| `test_coherence_uniform_volume` | Coherence == 1.0 for single-phase volume | Within 1e-5 |
| `test_coherence_random_volume` | Coherence < 1.0 for iid noise volume | Score < 1.0 |
| `test_tpb_density_non_negative` | TPB density is non-negative | Value >= 0 |
| `test_conditioning_accuracy_runs` | Accuracy test runs without crash | No NaN; R² <= 1.0 |
| `test_visualisation_saves` | PNG written to disk | `os.path.exists(save_path)` |

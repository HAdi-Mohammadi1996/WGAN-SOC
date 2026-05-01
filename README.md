# Conditional SliceGAN — Development Branch

> **Branch**: `dev`  
> **Status**: 🔧 In Development  
> **Python**: 3.10+  
> **Framework**: PyTorch 2.0+

---

## What This Project Does

This repository implements a **Conditional SliceGAN** for generating realistic 3D microstructure subvolumes from 2D training images. The user specifies target microstructural properties (**volume fraction** and **average pore size** per phase) and the model generates statistically consistent 3D volumes matching those specifications.

### Key capabilities to achieve

* Generate 3D microstructure volumes from a single representative 2D micrograph

* User-specified conditioning on volume fraction and average pore size per phase

* Supports n-phase segmented microstructures (2-phase, 3-phase)

* Full statistical validation framework against real microstructure data

---

## Architecture Choice: SliceGAN + Conditioning

SliceGAN (Kench & Cooper, 2021) resolves the 3D generation from 2D training data problem with three key contributions:

* A **slicing step** between the 3D generator and 2D discriminator, eliminating the need for 3D training data

* A **uniform information density** constraint on transpose convolution parameters, eliminating edge artefacts

* A **spatial latent input of size 4×4×4** rather than 1×1×1, enabling clean large-volume generation at inference

This repository adds:

* A **conditioning encoder** that maps user parameters to a spatial embedding concatenated with the latent noise

* An **auxiliary regression head** on the discriminator that enforces conditioning accuracy during training

* A **staged training curriculum** for stable convergence

---

## Development Phases

The project is developed in six sequential phases. **No phase begins until all tests for the previous phase pass.** 

---

### Phase 1 — Data Pipeline and Preprocessing

**Goal**: Build a robust, tested data loading pipeline for 3D microstructure volumes.

**What is built**:

* `MicrostructureDataset`: loads 3D volumetric data from `.mat` or `.tiff` stack files, converts to one-hot encoding, and extracts random 2D slices along x, y, z axes

* `compute_volume_fraction`: computes per-phase volume fractions from a one-hot volume tensor

* `compute_average_pore_size`: compute mean inscribed sphere radius per phase

* `compute_conditioning_vector`: assembles the full conditioning vector `[vf_1...vf_N, ps_1...ps_N]`

* Data augmentation: random 90° rotations and mirror flips applied consistently before slicing

**Tests — `tests/test_phase1.py`**:
 
| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_dataset_loads` | Dataset returns tensors of correct shape | Shape matches `[n_phases, 64, 64]` for slices |
| `test_onehot_valid` | One-hot encoding is correct | Channel sum == 1.0 at every voxel (tol 1e-4) |
| `test_volume_fraction_sums_to_one` | Phase fractions sum to 1 | `sum(vf) == 1.0` (tol 1e-4) |
| `test_pore_size_positive` | Pore size is physically meaningful | All values > 0 |
| `test_slice_shapes` | Slices along all three axes are correct | Shape `[n_phases, 64, 64]` for all axes |
| `test_conditioning_vector_length` | Conditioning vector has correct length | Length == `2 * n_phases` |
| `test_augmentation_preserves_vf` | Augmentation does not change volume fractions | Max VF change < 1e-4 across 10 augmented samples |
| `test_dataloader_iterates` | DataLoader runs without error | 2 full iterations with `batch_size=4, num_workers=2` |
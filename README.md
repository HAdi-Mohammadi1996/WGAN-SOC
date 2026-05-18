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

### Phase 2 — Model Architecture
 
**Goal**: Implement and verify the generator and discriminator architectures before any training.
 
**What is built**:
- `Generator3D`: 3D transpose convolutional generator with 5 layers, softmax output, and uniform information density enforced via the SliceGAN parameter rules
- `ConditioningEncoder`: dense encoder that maps the conditioning vector to a spatial embedding of shape `[batch, nz, 4, 4, 4]`
- `Discriminator2D`: 2D convolutional network with two output heads — a Wasserstein critic scalar and an auxiliary regression head predicting the conditioning vector
- Spatial conditioning utilities: functions to tile conditioning vectors to feature map spatial dimensions for concatenation
**The uniform information density rules** enforced in the generator:
- stride < kernel_size
- kernel_size % stride == 0
- padding ≥ kernel_size − stride
- Practical parameter sets: `{4,2,2}` for layers 1–4, `{4,2,3}` for layer 5
**The latent input shape `[batch, nz, 4, 4, 4]`**: Using spatial size 4 rather than 1 ensures the generator learns kernel overlap behaviour from the very first layer, which allows clean scaling to larger output volumes at inference without distortion.
 
**Tests — `tests/test_phase2.py`**:
 
| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_generator_output_shape` | Generator produces correct volume shape | `[batch, n_phases, 64, 64, 64]` |
| `test_softmax_valid` | Generator output is a valid probability distribution | Phase probabilities sum to 1 at every voxel (tol 1e-5) |
| `test_discriminator_output_shapes` | Both discriminator heads produce correct shapes | Critic: `[batch, 1]`, Regression: `[batch, 2*n_phases]` |
| `test_uniform_information_density` | Edge voxels have similar activation magnitude to centre | Edge/centre ratio within 10% |
| `test_conditioning_encoder_shape` | Encoder output has correct shape | `[batch, nz, 4, 4, 4]` |
| `test_generator_with_conditioning` | Conditioning integrates without shape errors | Forward pass completes, output shape unchanged |
| `test_discriminator_with_conditioning` | Conditioning channels append correctly | Forward pass completes with extra input channels |
| `test_parameter_counts` | Model sizes are logged | Runs without error, values printed to console |
| `test_large_volume_generation` | Generator scales to larger volumes at inference | `[batch, n_phases, 128, 128, 128]` from `z` of size `[batch, nz, 8, 8, 8]` |
 
All tests use random tensors as inputs — no data loading required

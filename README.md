# Conditional SliceGAN — Development Branch

> **Branch**: `dev`  
> **Status**: 🔧 In Development  
> **Python**: 3.10+  
> **Framework**: PyTorch 2.0+

---

## What This Project Does

This repository implements a **Conditional SliceGAN** for generating realistic 3D microstructure subvolumes from 2D training images. The user specifies target microstructural properties — **volume fraction** and **average pore size** per phase — and the model generates statistically consistent 3D volumes matching those specifications.

The system is built on the SliceGAN architecture (Kench & Cooper, 2021), which resolves the fundamental incompatibility between a 3D generator and 2D training data by slicing generated volumes along all three axes and evaluating each slice with a 2D discriminator. This repository extends that baseline with a conditional generation mechanism, allowing user-controlled microstructure synthesis.

### Key Capabilities
- Generates 3D microstructure volumes from a single representative 2D micrograph
- User-specified conditioning on volume fraction and average pore size per phase
- Supports n-phase segmented microstructures (2-phase, 3-phase, etc.)
- Handles both isotropic and anisotropic materials
- Post-training generation of arbitrarily large volumes in seconds
- Full statistical validation framework against real microstructure data

---

## Scientific Background

### The Problem
Obtaining real 3D microstructure data (e.g. via Xe plasma focused ion beam SEM) is expensive and produces limited sample volumes. Computational workflows for uncertainty quantification, design optimisation, and finite element simulation require large numbers of statistically representative microstructure realisations that cannot be obtained experimentally at scale.

### Why GAN-Based Generation
Traditional methods such as DREAM.3D use ellipsoid packing with user-defined geometric assumptions, which constrains topology and fails to reproduce properties like phase connectivity and triple-phase boundary density — the features that actually govern material performance. A GAN trained on real microstructure images learns the full statistical distribution without any geometric assumptions, and once trained can generate thousands of unique realistic volumes in seconds.

### Why Conditional
An unconditional GAN generates microstructures that look like the training data but gives the user no control over their properties. The conditional extension allows a user to specify desired bulk characteristics and receive microstructures consistent with those specifications — enabling targeted exploration of the microstructure design space.

### Architecture Choice: SliceGAN + Conditioning
SliceGAN (Kench & Cooper, 2021) resolves the 3D generation from 2D training data problem with three key contributions:
- A **slicing step** between the 3D generator and 2D discriminator, eliminating the need for 3D training data
- A **uniform information density** constraint on transpose convolution parameters, eliminating edge artefacts
- A **spatial latent input of size 4×4×4** rather than 1×1×1, enabling clean large-volume generation at inference

This repository adds:
- A **conditioning encoder** that maps user parameters to a spatial embedding concatenated with the latent noise
- An **auxiliary regression head** on the discriminator that enforces conditioning accuracy during training
- A **staged training curriculum** for stable convergence

---

## Repository Structure

```
conditional_slicegan/
├── data/
│   ├── dataset.py              # MicrostructureDataset — loading, slicing, augmentation
│   ├── preprocessing.py        # One-hot encoding, normalisation
│   └── metrics_data.py         # Volume fraction, average pore size computation
│
├── models/
│   ├── generator.py            # Generator3D — 3D transpose conv network
│   ├── discriminator.py        # Discriminator2D — 2D conv + critic + regression heads
│   └── conditioning.py         # ConditioningEncoder, spatial tiling utilities
│
├── training/
│   ├── trainer.py              # ConditionalSliceGANTrainer
│   ├── losses.py               # Wasserstein loss, gradient penalty, regression loss
│   └── curriculum.py           # Stage scheduling, lambda annealing
│
├── evaluation/
│   ├── metrics.py              # MicrostructureMetrics — full validation suite
│   ├── validation.py           # Statistical comparison, conditioning accuracy
│   └── visualisation.py        # Cross-section and 3D rendering utilities
│
├── tests/
│   ├── test_phase1.py          # Data pipeline tests
│   ├── test_phase2.py          # Architecture tests
│   ├── test_phase3.py          # Training loop tests
│   ├── test_phase4.py          # Conditioning integration tests
│   └── test_phase5.py          # Validation framework tests
│
├── config.py                   # Hyperparameter configuration
├── train.py                    # Main training entry point
├── generate.py                 # Inference entry point
├── validate.py                 # Standalone evaluation script
└── requirements.txt
```

---



## Status Tracker

| Phase | Status | Tests passing |
|-------|--------|--------------|
| Phase 1 — Data Pipeline | ✅Completed  | 8 / 8 |
| Phase 2 — Architecture | ⬜ Not started | 0 / 9 |
| Phase 3 — Training Loop | ⬜ Not started | 0 / 8 |
| Phase 4 — Conditioning | ⬜ Not started | 0 / 8 |
| Phase 5 — Validation | ⬜ Not started | 0 / 9 |
| Phase 6 — Integration | ⬜ Not started | 0 / 4 |

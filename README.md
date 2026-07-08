# Phase 4: Conditioning Integration

**Goal**: Extend the unconditional SliceGAN training loop with per-phase conditioning on volume
 fraction (VF) and average pore size (PS).

 **What changes vs Phase 3 (`_training_loop`)**:
 - Generator step: conditioning vector is encoded via `ConditioningEncoder` → embedded maps are
   cat-ed with z before layer 1.
 - Discriminator step: conditioning vector is tiled to 2D spatial maps and appended as extra input
   channels to both real and fake slices.
 - Regression loss: `MSE(D_regression(real_slice), true_cond_vector)` is added to the discriminator
   loss, weighted by `lambda_reg`.
 - Curriculum: `lambda_reg` is annealed over three stages (see table below) to prevent premature
   collapse.

 **Staged curriculum**:

 | Stage | Epochs   | lambda_reg          | Effect                                 |
 |-------|----------|---------------------|----------------------------------------|
 | 1     | 1–50     | 0.0                 | Identical to unconditional baseline    |
 | 2     | 51–150   | 0.1 → 1.0 (linear) | Conditioning warmup                    |
 | 3     | 151+     | 1.0                 | Full conditioning                      |

 **Conditioning vector format** (unchanged from Phase 1):
 `[vf_0, ..., vf_{N-1}, ps_0, ..., ps_{N-1}]` — length `2 * n_phases`.
 Normalised to zero-mean/unit-std using stats computed by `compute_dataset_conditioning_stats`.

 **New files**:
- `slicegan/curriculum.py` — `get_lambda_reg(epoch, ...)`, `LambdaSchedule` class
- `tests/test_phase4.py` — 10 tests

**Modified files**:
 - `slicegan/model.py` — add `train_conditional()`, `sample_conditioning_from_batch()`,
   `generate_conditioned_samples()`

 **Tests — `tests/test_phase4.py`** (all must pass before proceeding):

 | Test | What it checks | Pass criterion |
 |------|---------------|----------------|
 | `test_generator_input_shape_with_conditioning` | Generator input includes conditioning embedding | `[batch, nz+embed_dim, 4, 4, 4]` |
 | `test_discriminator_input_channels` | Conditioning channels appended to slices | In-channels == `n_phases + conditioning_dim` |
 | `test_regression_loss_zero_on_perfect` | Loss == 0 when pred equals target | Loss < 1e-6 |
 | `test_regression_loss_decreases` | Regression loss improves over 200 steps | Loss at step 200 < step 0 |
 | `test_stage1_lambda_reg_is_zero` | Stage 1 returns `lambda_reg=0` | `get_lambda_reg(epoch ≤ 50) == 0.0` |
 | `test_stage2_lambda_schedule` | Lambda increases linearly in stage 2 | Values match expected schedule |
 | `test_conditioning_normalisation` | Normalisation uses dataset stats | Output is z-scored correctly |
 | `test_conditioned_generation_shape` | `generate_conditioned_samples` output shape | `[n_samples, n_phases, 64, 64, 64]` |
 | `test_conditioning_rough_accuracy` | Short training shifts VF toward target | VF within 20 % relative error of target |
 | `test_no_nan_losses` | Numerical stability | Zero NaN losses over 500 iterations |
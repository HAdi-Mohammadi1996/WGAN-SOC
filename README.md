
---

### Phase 3: Training Loop (Unconditional Baseline)

**Goal**: Replicate  SliceGAN exactly as the unconditional baseline before adding any conditioning.

**Background — exact hyperparameters from SliceGAN `model.py`**:

```python
num_epochs    = 100
batch_size    = 8        # generator batch size
D_batch_size  = 8        # discriminator batch size
lrg           = 0.0001
lrd           = 0.0001
beta1         = 0.9      # NOTE: repo uses 0.9
beta2         = 0.99     # NOTE: repo uses 0.99
Lambda        = 10       # gradient penalty coefficient
critic_iters  = 5        # NOTE: discriminator steps per generator step — NOT 1
lz            = 4        # latent spatial size
workers       = 0        # DataLoader workers
```

**Training loop structure from `model.py`**:

- Outer loop over epochs; inner loop zips the three axis dataloaders: `for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz))`
- Each iteration: run `critic_iters=5` discriminator updates, then 1 generator update when `i % critic_iters == 0`
- Discriminator update: generate fake volume, apply permute+reshape slicing, compute `out_fake - out_real + gradient_penalty`, call `disc_cost.backward()` + `optimizer.step()`
- Generator update: generate new fake volume, slice, compute `-mean(D(fake))` summed over axes, call `errG.backward()` + `optG.step()`
- Checkpoint + sample images saved every 25 iterations via `torch.save` + `util.test_plotter`
- Logs: `disc_real_log`, `disc_fake_log`, `gp_log`, `Wass_log` per iteration

**Tasks**:

1. Implement `util.calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, Lambda, nc)` matching the SliceGAN `util.py` function signature

2. Implement `util.test_plotter(img, n_slices, imtype, path)` for saving sample slice images

3. Implement `util.calc_eta(steps, time_now, start, i, epoch, num_epochs)` for ETA logging

4. Implement `model.train(pth, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf)` matching the SliceGAN `model.py` function signature exactly, including:
   - Three separate DataLoaders for x, y, z axes from `dataset_xyz`
   - Isotropic check: `if len(real_data) == 1: real_data *= 3; isotropic = True`
   - One discriminator (isotropic) or three discriminators (anisotropic)
   - The exact permutation slicing with `d1,d2,d3` tuples `[2,3,4],[3,2,2],[4,4,3]`
   - Save checkpoint to `pth + '_Gen.pt'` and `pth + '_Disc.pt'` every 25 iterations

**Tests — `tests/test_phase3.py`** (all must pass before proceeding):

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_slice_permutation_shapes` | Slicing produces correct 2D shapes for all axes | `[l*batch, nc, l, l]` for each axis |
| `test_slice_permutation_axes_distinct` | Three axis slicings are genuinely different views | Mean pixel values differ across axes |
| `test_gradient_penalty_positive` | GP is positive and finite | GP > 0 and not NaN |
| `test_discriminator_update_runs` | One discriminator step completes | Scalar loss returned, no error |
| `test_generator_update_runs` | One generator step completes | Scalar loss returned, no error |
| `test_critic_iters_respected` | Generator only updates every 5 iterations | Generator params unchanged after 4 steps, changed after 5 |
| `test_all_four_logs_populated` | All four log lists populated | `disc_real_log`, `disc_fake_log`, `gp_log`, `Wass_log` all non-empty |
| `test_checkpoint_naming` | Checkpoint saved with correct filenames | `pth + '_Gen.pt'` and `pth + '_Disc.pt'` exist |
| `test_checkpoint_reproducibility` | Reloaded generator produces identical output | Max abs diff < 1e-6 for same seed |
| `test_isotropic_shares_discriminator` | Isotropic mode reuses `netDs[0]` | Single discriminator updated for all axes |
| `test_no_nan_losses` | Training is numerically stable | Zero NaN losses over 100 iterations |
| `test_wasserstein_distance_improves` | Training makes progress | Mean Wass distance over last 10 steps < first 10 steps |
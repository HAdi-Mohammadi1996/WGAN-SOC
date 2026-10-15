# Phase 6: Integration and Configuration

**Goal**: Wrap the Phase 1–5 components into clean command-line entry points that follow the style of
SliceGAN's `run_slicegan.py`, extended for conditional training, conditioned inference and standalone
validation. All hyperparameters move into a single `config.yaml`.

**New files**:
- `config.yaml` — every data, architecture, training, conditioning and logging parameter (SliceGAN naming)
- `slicegan/config.py` — `load_config` / `validate_config` (YAML → dict with defaults and consistency checks)
- `run_slicegan.py` — main entry point: `1` = train, `0` = generate a `.tif` sample
- `generate.py` — conditioned inference from target VF and pore size
- `validate.py` — standalone evaluation producing a CSV and PNG report
- `tests/test_phase6.py` — integration tests on synthetic `.mat` data

**Changes to existing code**:
- `slicegan/model.py` — `train` / `train_conditional` accept a `cfg` dict so the hyperparameters
  (`num_epochs`, `batch_size`, `lrg`, `lrd`, `beta1`, `beta2`, `Lambda`, `critic_iters`, `save_interval`)
  override the module-level SliceGAN defaults, which stay as fallbacks.
- `slicegan/model.py` — conditioning normalisation stats `(mean, std)` are saved next to the checkpoint
  (`{project_path}_norm.npz`), so inference doesn't need the training data.

**`config.yaml`** (parameter names follow the SliceGAN repo):

| Group | Keys | Notes |
|-------|------|-------|
| Data | `data_path`, `data_type`, `image_type`, `img_size`, `img_channels`, `scale_factor` | A single `data_path` entry means isotropic (`real_data *= 3`) |
| Subvolumes (new) | `subvol_size`, `subvol_stride`, `min_phase_coverage` | `subvol_size` must equal `img_size`, otherwise it raises an error |
| Architecture | `z_channels`, `lz`, `gk/gs/gf/gp`, `dk/ds/df/dp` | Defaults are SliceGAN Table 1 |
| Training | `num_epochs`, `batch_size`, `D_batch_size`, `lrg`, `lrd`, `beta1`, `beta2`, `Lambda`, `critic_iters` | SliceGAN `model.py` values (`beta1=0.9`, `beta2=0.99`, `critic_iters=5`) |
| Conditioning | `conditioning_dim`, `conditioning_hidden`, `lambda_reg`, `stage1_epochs`, `stage2_epochs`, `lambda_reg_warmup_start` | `conditioning_dim` must equal `2 * img_channels`; stage keys build the `LambdaSchedule` |
| Logging | `save_interval`, `project_path` | Checkpoints are written to `{project_path}_Gen.pt` and `{project_path}_Disc.pt` |

**Entry points**:

| Command | Behaviour | Output |
|---------|-----------|--------|
| `python run_slicegan.py 1 [--config config.yaml]` | Trains unconditionally (`model.train`) or conditionally (`model.train_conditional`), depending on the config | `{project_path}_Gen.pt`, `_Disc.pt`, `_CondEnc.pt`, `_norm.npz`, sample slices |
| `python run_slicegan.py 0 [--config config.yaml]` | Loads the generator and produces one volume, argmax-segmented | `{project_path}.tif` (SliceGAN output format) |
| `python generate.py --checkpoint runs/exp1 --vf 0.30 0.42 0.28 --pore_size 2.5 3.1 1.8 --n_samples 20` | Normalises the target with the saved stats, then calls `generate_conditioned_samples` | `.npy` + `.tif` per sample, shape `[n_samples, 64, 64, 64]` |
| `python validate.py --checkpoint runs/exp1 --config config.yaml [--out report/]` | Generates volumes and compares them with real subvolumes | `metrics.csv`, box plots, scatter plots, cross-sections |

**`validate.py` report** (built on the Phase 5 functions in `slicegan/metrics.py`):

| Section | Source function | Output |
|---------|-----------------|--------|
| Statistical similarity | `statistical_similarity_test(real, gen, save_path=...)` | KS statistic/p-value per metric → CSV; `boxplots.png` |
| Conditioning accuracy | `conditioning_accuracy_test(..., save_dir=...)` | R² and MAE for VF and pore size → CSV; `vf_scatter.png`, `ps_scatter.png` |
| Edge quality | `edge_centre_consistency(volume, shell=4, tol=0.2)` | Pass rate across generated volumes → CSV |
| Coherence | `MicrostructureMetrics.inter_slice_coherence` | Mean ± std → CSV (target > 0.70) |
| Visuals | `visualise_volume` | Orthogonal cross-sections, real vs. generated |

**Success criteria** (from the project brief, checked by `validate.py` on a trained model):
- R² ≥ 0.85 between target and generated volume fractions
- KS p > 0.05 against real data distributions
- Inter-slice coherence > 0.70
- 10–100 unique volumes generated in seconds after training

**Deferred from Phase 5**: tortuosity, relative diffusivity and formation factor need a steady-state
diffusion solver (e.g. `taufactor`). They're optional extensions to the `validate.py` report.

**Tests — `tests/test_phase6.py`** (all must pass; synthetic data only, tiny architecture, CPU):

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_config_loads_defaults` | `config.yaml` parses and every required key is present | `load_config` returns a dict with SliceGAN default values |
| `test_config_subvol_mismatch_raises` | `subvol_size != img_size` is rejected | `ValueError` with a descriptive message |
| `test_config_conditioning_dim_mismatch_raises` | `conditioning_dim != 2 * img_channels` is rejected | `ValueError` |
| `test_run_slicegan_train_5_epochs` | `run_slicegan.py 1` trains 5 epochs on a synthetic `.mat` | Exit code 0; `_Gen.pt` and `_Disc.pt` exist |
| `test_run_slicegan_generate_tif` | `run_slicegan.py 0` writes a SliceGAN-format `.tif` | File exists; 3-D integer volume of the expected size |
| `test_norm_stats_saved_with_checkpoint` | Conditional training saves the normalisation stats | `_norm.npz` exists and holds `mean`, `std` of length `2 * nc` |
| `test_generate_conditioned_shape` | `generate.py` produces conditioned volumes | `.npy` shape `[n_samples, l, l, l]`; `.tif` written |
| `test_validate_report_complete` | `validate.py` produces the full report | CSV has all metric rows; every PNG exists |
| `test_full_pipeline_no_intervention` | train → generate → validate chained in one run | All three steps exit 0 with no manual input |

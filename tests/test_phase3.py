"""Phase 3 tests: unconditional training loop.

Architecture:
  NC=2, NZ=4, LZ=4, L=8
  Generator  — 1 layer ConvTranspose3d(NZ, NC, k=5, s=1, p=0) + Softmax → [batch, 2, 8, 8, 8]
  Discriminator — 1 layer Conv2d(NC, 16, k=8, s=1, p=0) → [batch, 16, 1, 1] → Linear(16,1)
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import TensorDataset

from slicegan.networks import Generator3D, Discriminator2D
from slicegan.util import calc_gradient_penalty
from slicegan.model import _training_loop, _D1, _D2, _D3

# ── Constants ────────────────────────────────────────────────────────────────
NC  = 2    # phases
NZ  = 4    # latent channels
LZ  = 4    # latent spatial size
L   = 8    # output spatial size (8³ volumes, 8×8 slices)

# Single-layer architecture (fast; used for most tests)
# Generator: ConvTranspose3d(4→2, k=5, s=1, p=0) → spatial: (4-1)*1+5=8 ✓
GK = [5];  GS = [1];  GF = [NC];  GP_ = [0]

# Discriminator: Conv2d(2→16, k=8, s=1, p=0) → spatial: (8-8)/1+1=1 ✓
DK = [8];  DS = [1];  DF = [16];  DP_ = [0]

# Two-layer architecture (used for the Wasserstein-convergence test)
# Generator: [NZ,4,4,4] → conv(k=3,s=1) → [16,6,6,6] → conv(k=3,s=1) → [NC,8,8,8]
GK2 = [3, 3];  GS2 = [1, 1];  GF2 = [16, NC];  GP2 = [0, 0]
# Discriminator: [NC,8,8] → conv(k=4,s=2,p=1) → [16,4,4] → conv(k=4,s=1) → [32,1,1]
DK2 = [4, 4];  DS2 = [2, 1];  DF2 = [16, 32];  DP2 = [1, 0]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_G():
    return Generator3D(nz=NZ, n_phases=NC, gk=GK, gs=GS, gf=GF, gp=GP_)


def _make_D():
    return Discriminator2D(n_phases=NC, n_conditions=0, dk=DK, ds=DS, df=DF, dp=DP_)


def _make_datasets(n=64):
    """Synthetic [TensorDataset_x, TensorDataset_y, TensorDataset_z] of shape [n, NC, L, L].

    Patches are softmax-normalised so channels sum to 1 — matching the generator's
    output structure and avoiding trivial discriminator shortcuts.
    """
    raw = torch.randn(n, NC, L, L)
    patches = torch.softmax(raw, dim=1)
    ds = TensorDataset(patches)
    return [ds, ds, ds]


def _make_uniform_datasets(n=64):
    """Real data = uniform [1/NC, ..., 1/NC] softmax patches.

    Each pixel has equal probability across all NC phases. A 1-layer generator
    can quickly learn to produce this (equal logits → equal softmax).  Once G
    matches the real distribution D cannot distinguish them and the Wasserstein
    distance drops, making the test reliably directional.
    """
    patches = torch.ones(n, NC, L, L) / NC
    ds = TensorDataset(patches)
    return [ds, ds, ds]


def _make_G2():
    return Generator3D(nz=NZ, n_phases=NC, gk=GK2, gs=GS2, gf=GF2, gp=GP2)


def _make_D2():
    return Discriminator2D(n_phases=NC, n_conditions=0, dk=DK2, ds=DS2, df=DF2, dp=DP2)


def _make_ref_datasets(n_vols=32):
    """Real data = 2D slices from a fixed-seed reference generator (same 2-layer arch).

    Using G_ref outputs as real data ensures the training G has exact capacity to match
    the real distribution; Wasserstein distance reliably decreases as G converges.
    """
    torch.manual_seed(100)
    G_ref = _make_G2()
    G_ref.eval()
    patches = []
    with torch.no_grad():
        for _ in range(n_vols):
            z = torch.randn(1, NZ, LZ, LZ, LZ)
            vol = G_ref(z)[0]           # [NC, L, L, L]
            patches.append(vol[:, L // 2, :, :])   # x mid-slice
            patches.append(vol[:, :, L // 2, :])   # y mid-slice
    patches = torch.stack(patches)      # [2*n_vols, NC, L, L]
    ds = TensorDataset(patches)
    return [ds, ds, ds]


def _params_snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _params_equal(s1, s2):
    return all(torch.allclose(a, b) for a, b in zip(s1, s2))


def _run_loop(G, D, datasets, tmp_path, *, epochs=1, batch=2):
    pth = str(tmp_path / 'ckpt')
    return _training_loop(
        datasets, G, D, NC, L, NZ, pth=pth,
        isotropic=True,
        num_epochs=epochs,
        batch_size=batch,
        D_batch_size=batch,
        critic_iters=5,
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_slice_permutation_shapes():
    batch = 3
    fake = torch.randn(batch, NC, L, L, L)
    for j in range(3):
        sliced = fake.permute(0, _D1[j], 1, _D2[j], _D3[j]).reshape(batch * L, NC, L, L)
        assert sliced.shape == (batch * L, NC, L, L), \
            f"Axis {j}: expected {(batch*L, NC, L, L)}, got {tuple(sliced.shape)}"


def test_slice_permutation_axes_distinct():
    torch.manual_seed(0)
    fake = torch.randn(2, NC, L, L, L)
    slices = [
        fake.permute(0, _D1[j], 1, _D2[j], _D3[j]).reshape(2 * L, NC, L, L)
        for j in range(3)
    ]
    assert not torch.equal(slices[0], slices[1]), "Axis 0 and 1 slicings must differ"
    assert not torch.equal(slices[1], slices[2]), "Axis 1 and 2 slicings must differ"


def test_gradient_penalty_positive():
    D = _make_D()
    n_fake = 2 * L   # = 16, matching batch_size * l in training
    real = torch.rand(n_fake, NC, L, L)
    fake = torch.rand(n_fake, NC, L, L)
    gp = calc_gradient_penalty(D, real, fake, n_fake, L, torch.device('cpu'), Lambda=10.0, nc=NC)
    assert torch.isfinite(gp), "GP must be finite"
    assert gp.item() > 0, "GP must be positive"


def test_discriminator_update_runs():
    G = _make_G()
    D = _make_D()
    optD = torch.optim.Adam(D.parameters(), lr=1e-4)

    z = torch.randn(2, NZ, LZ, LZ, LZ)
    with torch.no_grad():
        fake_vol = G(z)

    j = 0
    fake_slice = fake_vol.permute(0, _D1[j], 1, _D2[j], _D3[j]).reshape(2 * L, NC, L, L)
    real_slice = torch.rand(2 * L, NC, L, L)

    D.zero_grad()
    out_real, _ = D(real_slice)
    out_fake, _ = D(fake_slice.detach())
    gp = calc_gradient_penalty(D, real_slice, fake_slice.detach(), 2 * L, L,
                               torch.device('cpu'), 10.0, NC)
    loss = out_fake.mean() - out_real.mean() + gp
    loss.backward()
    optD.step()

    assert loss.dim() == 0, "D loss must be a scalar"
    assert torch.isfinite(loss), "D loss must be finite"


def test_generator_update_runs():
    G = _make_G()
    D = _make_D()
    optG = torch.optim.Adam(G.parameters(), lr=1e-4)

    G.zero_grad()
    z = torch.randn(2, NZ, LZ, LZ, LZ)
    fake_vol = G(z)
    errG = sum(
        -D(fake_vol.permute(0, _D1[j], 1, _D2[j], _D3[j]).reshape(2 * L, NC, L, L))[0].mean()
        for j in range(3)
    )
    errG.backward()
    optG.step()

    assert errG.dim() == 0, "G loss must be a scalar"
    assert torch.isfinite(errG), "G loss must be finite"


def test_critic_iters_respected():
    """Generator updates only at i % critic_iters == 0 (i.e. at i=0, 5, 10, ...)."""
    torch.manual_seed(1)
    G = _make_G()
    D = _make_D()
    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.99))

    critic_iters = 5
    batch = 2
    real = torch.rand(batch * L, NC, L, L)

    def _step(i):
        # D update
        with torch.no_grad():
            z = torch.randn(batch, NZ, LZ, LZ, LZ)
            fv = G(z)
        for j in range(3):
            fs = fv.permute(0, _D1[j], 1, _D2[j], _D3[j]).reshape(batch * L, NC, L, L)
            D.zero_grad()
            out_r, _ = D(real)
            out_f, _ = D(fs.detach())
            gp = calc_gradient_penalty(D, real, fs.detach(), batch * L, L,
                                       torch.device('cpu'), 10.0, NC)
            (out_f.mean() - out_r.mean() + gp).backward()
            optD.step()

        # G update only when i % critic_iters == 0
        if i % critic_iters == 0:
            G.zero_grad()
            z2 = torch.randn(batch, NZ, LZ, LZ, LZ)
            fv2 = G(z2)
            errG = sum(
                -D(fv2.permute(0, _D1[j], 1, _D2[j], _D3[j]).reshape(batch * L, NC, L, L))[0].mean()
                for j in range(3)
            )
            errG.backward()
            optG.step()

    _step(0)   # G updates (0 % 5 == 0)
    snap_after_0 = _params_snapshot(G)

    for i in range(1, 5):
        _step(i)  # G should NOT update

    assert _params_equal(snap_after_0, _params_snapshot(G)), \
        "Generator params must not change at steps i=1 through i=4"

    _step(5)    # G updates again (5 % 5 == 0)
    assert not _params_equal(snap_after_0, _params_snapshot(G)), \
        "Generator params must change at step i=5"


def test_all_four_logs_populated(tmp_path):
    G = _make_G()
    D = _make_D()
    logs = _run_loop(G, D, _make_datasets(32), tmp_path, epochs=1, batch=2)

    for key in ('disc_real_log', 'disc_fake_log', 'gp_log', 'Wass_log'):
        assert len(logs[key]) >= 1, f"'{key}' is empty after training"


def test_checkpoint_naming(tmp_path):
    G = _make_G()
    D = _make_D()
    _run_loop(G, D, _make_datasets(32), tmp_path, epochs=1, batch=2)

    pth = str(tmp_path / 'ckpt')
    assert os.path.isfile(pth + '_Gen.pt'),  f"Missing {pth}_Gen.pt"
    assert os.path.isfile(pth + '_Disc.pt'), f"Missing {pth}_Disc.pt"


def test_checkpoint_reproducibility(tmp_path):
    """Save G → load G → same z → identical output."""
    G = _make_G()
    D = _make_D()
    _run_loop(G, D, _make_datasets(32), tmp_path, epochs=1, batch=2)

    save_path = str(tmp_path / 'final_Gen.pt')
    torch.save(G, save_path)

    # Load to CPU for comparison
    G_cpu = G.to('cpu')
    G_loaded = torch.load(save_path, map_location='cpu', weights_only=False).to('cpu')
    G_cpu.eval()
    G_loaded.eval()

    torch.manual_seed(99)
    z = torch.randn(1, NZ, LZ, LZ, LZ)
    with torch.no_grad():
        out1 = G_cpu(z)
        out2 = G_loaded(z)

    max_diff = (out1 - out2).abs().max().item()
    assert max_diff < 1e-6, f"Reloaded G differs by {max_diff}"


def test_isotropic_shares_discriminator(tmp_path):
    """In isotropic mode, netDs list must hold the same object for all three axes."""
    G = _make_G()
    D = _make_D()
    result = _run_loop(G, D, _make_datasets(32), tmp_path, epochs=1, batch=2)

    netDs = result['netDs']
    assert id(netDs[0]) == id(netDs[1]) == id(netDs[2]), \
        "Isotropic mode must reuse a single Discriminator instance for all three axes"


def test_no_nan_losses(tmp_path):
    """No NaN losses over ~200 training iterations."""
    torch.manual_seed(0)
    G = _make_G()
    D = _make_D()
    # 32 samples / batch_size 2 = 16 steps/epoch × 13 epochs ≈ 208 steps
    logs = _training_loop(
        _make_datasets(32), G, D, NC, L, NZ, pth='',
        isotropic=True, num_epochs=13, batch_size=2, D_batch_size=2, critic_iters=5,
    )

    for key in ('disc_real_log', 'disc_fake_log', 'gp_log', 'Wass_log'):
        nan_count = sum(1 for v in logs[key] if v != v)
        assert nan_count == 0, f"'{key}' contains {nan_count} NaN value(s)"


def test_wasserstein_distance_improves(tmp_path):
    """Training makes progress: mean Wasserstein distance decreases from start to end.

    Strategy: pre-train the discriminator (D-only) until it clearly discriminates
    real uniform patches from G's random output, establishing a high initial Wass.
    Then run full training (D + G) and verify that G's improvement pulls Wass down.

    Uses a two-layer G / D so the discriminator can detect spatial patterns.
    """
    torch.manual_seed(42)
    G = _make_G2()
    D = _make_D2()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    real = (torch.ones(2 * L, NC, L, L) / NC).to(device)   # uniform [0.5, 0.5]
    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # Phase 1: pre-train D (500 outer steps × 3 axes) so first10 is high
    for _ in range(500):
        with torch.no_grad():
            z = torch.randn(2, NZ, LZ, LZ, LZ, device=device)
            fv = G(z)
        for j in range(3):
            fs = fv.permute(0, _D1[j], 1, _D2[j], _D3[j]).reshape(2 * L, NC, L, L)
            D.zero_grad()
            out_r, _ = D(real)
            out_f, _ = D(fs.detach())
            gp = calc_gradient_penalty(D, real, fs.detach(), 2 * L, L, device, 10.0, NC)
            (out_f.mean() - out_r.mean() + gp).backward()
            optD.step()

    # Phase 2: full training — G now competes against the discriminating D
    n = 64
    patches = (torch.ones(n, NC, L, L) / NC)   # keep on CPU; _training_loop moves to device
    ds = torch.utils.data.TensorDataset(patches)
    logs = _training_loop(
        [ds, ds, ds], G, D, NC, L, NZ, pth='',
        isotropic=True, num_epochs=40, batch_size=2, D_batch_size=2, critic_iters=5,
    )

    wass = logs['Wass_log']
    assert len(wass) >= 20, f"Need ≥ 20 Wass entries, got {len(wass)}"

    first10 = sum(wass[:10]) / 10
    last10  = sum(wass[-10:]) / 10
    assert last10 < first10, (
        f"Wasserstein distance should decrease; "
        f"first10={first10:.4f}, last10={last10:.4f}"
    )


# ── Standalone runner ─────────────────────────────────────────────────────────

def main():
    import pytest
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    main()

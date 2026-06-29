import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from . import preprocessing
from . import util
from .util import calc_gradient_penalty

# ── SliceGAN exact hyperparameters ──────────────────────────────────────────
_NUM_EPOCHS  = 100
_BATCH_SIZE  = 8
_D_BATCH_SIZE = 8
_LRG         = 1e-4
_LRD         = 1e-4
_BETA1       = 0.9
_BETA2       = 0.99
_LAMBDA      = 10
_CRITIC_ITERS = 5
_LZ          = 4
_WORKERS     = 0

# Default SliceGAN Table 1 architecture
_GK = [4, 4, 4, 4, 4]
_GS = [2, 2, 2, 2, 2]
_GF = [512, 256, 128, 64, 3]   # last entry overridden to nc in train()
_GP = [2, 2, 2, 2, 3]
_DK = [4, 4, 4, 4, 4]
_DS = [2, 2, 2, 2, 2]
_DF = [64, 128, 256, 512, 512]
_DP = [1, 1, 1, 1, 0]

# Exact SliceGAN permutation tuples: dim j uses (d1[j], d2[j], d3[j])
# Axis 0 → permute(0,2,1,3,4), Axis 1 → permute(0,3,1,2,4), Axis 2 → permute(0,4,1,2,3)
_D1 = [2, 3, 4]
_D2 = [3, 2, 2]
_D3 = [4, 4, 3]


def _training_loop(
    datasets,
    netG,
    netD,
    nc,
    l,
    nz,
    pth,
    isotropic=True,
    imtype='nphase',
    num_epochs=_NUM_EPOCHS,
    batch_size=_BATCH_SIZE,
    D_batch_size=_D_BATCH_SIZE,
    lrg=_LRG,
    lrd=_LRD,
    beta1=_BETA1,
    beta2=_BETA2,
    Lambda=_LAMBDA,
    critic_iters=_CRITIC_ITERS,
    lz=_LZ,
    workers=_WORKERS,
):
    """Core WGAN-GP training loop (unconditional baseline).

    Parameters
    ----------
    datasets : list of TensorDataset
        [dataset_x, dataset_y, dataset_z] — one per slicing axis.
    netG : Generator3D
        Pre-instantiated generator.
    netD : Discriminator2D or list of three Discriminator2D
        Single instance for isotropic; list of three for anisotropic.
    nc : int
        Number of phases (= output channels).
    l : int
        Spatial size of generated volumes and training patches.
    nz : int
        Latent noise channels.
    pth : str
        Checkpoint path prefix. Empty string disables checkpointing.
    isotropic : bool
        If True, the same netD is reused for all three axes.
    imtype : str
        Passed to util.test_plotter.
    num_epochs, batch_size, D_batch_size, lrg, lrd, beta1, beta2, Lambda,
    critic_iters, lz, workers : training hyperparameters.

    Returns
    -------
    dict with keys:
        disc_real_log, disc_fake_log, gp_log, Wass_log : lists of floats
        netDs : list of three discriminator references (first two are identical
                when isotropic=True)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = netG.to(device)

    if isotropic:
        single_D = netD.to(device) if not isinstance(netD, (list, tuple)) else netD[0].to(device)
        netDs = [single_D, single_D, single_D]
        single_opt = Adam(single_D.parameters(), lr=lrd, betas=(beta1, beta2))
        optDs = [single_opt, single_opt, single_opt]
    else:
        netD_list = netD if isinstance(netD, (list, tuple)) else [netD, netD, netD]
        netDs = [d.to(device) for d in netD_list]
        optDs = [Adam(d.parameters(), lr=lrd, betas=(beta1, beta2)) for d in netDs]

    optG = Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    loaderx = DataLoader(
        datasets[0], batch_size=D_batch_size, shuffle=True,
        num_workers=workers, drop_last=True,
    )
    loadery = DataLoader(
        datasets[1], batch_size=D_batch_size, shuffle=True,
        num_workers=workers, drop_last=True,
    )
    loaderz = DataLoader(
        datasets[2], batch_size=D_batch_size, shuffle=True,
        num_workers=workers, drop_last=True,
    )

    disc_real_log: list = []
    disc_fake_log: list = []
    gp_log: list = []
    Wass_log: list = []

    steps = min(len(loaderx), len(loadery), len(loaderz))
    start = time.time()
    fake_vol = None

    for epoch in range(num_epochs):
        for i, (bx, by, bz) in enumerate(zip(loaderx, loadery, loaderz)):
            real_batches = [bx[0].to(device), by[0].to(device), bz[0].to(device)]

            # ── Discriminator update ─────────────────────────────────────────
            with torch.no_grad():
                z = torch.randn(batch_size, nz, lz, lz, lz, device=device)
                fake_vol = netG(z)

            disc_real_acc = 0.0
            disc_fake_acc = 0.0
            gp_acc = 0.0

            for j in range(3):
                real_slice = real_batches[j]   # [D_batch, nc, l, l]
                fake_slice = (
                    fake_vol
                    .permute(0, _D1[j], 1, _D2[j], _D3[j])
                    .reshape(batch_size * l, nc, l, l)
                )

                # Tile real to match the larger fake slice count
                n_fake = batch_size * l
                tile = (n_fake + D_batch_size - 1) // D_batch_size
                real_tiled = real_slice.repeat(tile, 1, 1, 1)[:n_fake]

                netDs[j].zero_grad()
                out_real, _ = netDs[j](real_tiled)
                out_fake, _ = netDs[j](fake_slice.detach())
                gp = calc_gradient_penalty(
                    netDs[j], real_tiled, fake_slice.detach(),
                    n_fake, l, device, Lambda, nc,
                )
                disc_cost = out_fake.mean() - out_real.mean() + gp
                disc_cost.backward()
                optDs[j].step()

                disc_real_acc += out_real.mean().item()
                disc_fake_acc += out_fake.mean().item()
                gp_acc += gp.item()

            disc_real_log.append(disc_real_acc / 3)
            disc_fake_log.append(disc_fake_acc / 3)
            gp_log.append(gp_acc / 3)
            Wass_log.append((disc_real_acc - disc_fake_acc) / 3)

            # ── Generator update (every critic_iters steps) ──────────────────
            if i % critic_iters == 0:
                netG.zero_grad()
                z = torch.randn(batch_size, nz, lz, lz, lz, device=device)
                fake_vol_g = netG(z)
                errG = sum(
                    -netDs[j](
                        fake_vol_g
                        .permute(0, _D1[j], 1, _D2[j], _D3[j])
                        .reshape(batch_size * l, nc, l, l)
                    )[0].mean()
                    for j in range(3)
                )
                errG.backward()
                optG.step()
                fake_vol = fake_vol_g.detach()

            # ── Checkpoint (every 25 inner steps, including step 0) ──────────
            if i % 25 == 0 and pth:
                torch.save(netG, pth + '_Gen.pt')
                torch.save(netDs[0], pth + '_Disc.pt')
                util.test_plotter(fake_vol.cpu(), 8, imtype, pth)
                util.calc_eta(steps, time.time(), start, i, epoch, num_epochs)

    return {
        'disc_real_log': disc_real_log,
        'disc_fake_log': disc_fake_log,
        'gp_log': gp_log,
        'Wass_log': Wass_log,
        'netDs': netDs,
    }


def train(pth, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf):
    """Public entry point matching the SliceGAN model.train signature.

    Parameters
    ----------
    pth : str
        Checkpoint path prefix.
    imtype : str
        Image type string (e.g. 'nphase').
    datatype : str
        Data type passed to preprocessing.batch (e.g. 'mat3D', 'tif3D').
    real_data : list of str
        File paths. Length 1 for isotropic, length 3 for anisotropic.
    Disc : class
        Discriminator2D constructor.
    Gen : class
        Generator3D constructor.
    nc : int
        Number of phases.
    l : int
        Spatial size of training patches and generated volumes.
    nz : int
        Latent noise channels.
    sf : int
        Scale factor for spatial downsampling before patching.
    """
    pass
import time
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from . import preprocessing
from . import util
from .util import calc_gradient_penalty
from .conditioning import prepare_conditioning_maps_2d
from .curriculum import get_lambda_reg, LambdaSchedule

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
    isotropic = len(real_data) == 1
    if isotropic:
        real_data = real_data * 3

    gf = list(_GF)
    gf[-1] = nc

    datasets = preprocessing.batch(real_data, datatype, l, sf)

    netG = Gen(nz=nz, n_phases=nc, gk=_GK, gs=_GS, gf=gf, gp=_GP)
    netD = Disc(n_phases=nc, n_conditions=0, dk=_DK, ds=_DS, df=_DF, dp=_DP)

    _training_loop(
        datasets, netG, netD, nc, l, nz, pth,
        isotropic=isotropic, imtype=imtype,
    )


# ── Phase 4: Conditioning Integration ───────────────────────────────────────


def sample_conditioning_from_batch(patches, norm_stats, device):
    """Compute and normalise conditioning vectors from a batch of 2D patches.

    Parameters
    ----------
    patches : torch.Tensor
        Shape ``[batch, nc, l, l]`` — one-hot encoded 2D patches (float).
    norm_stats : tuple of (mean, std)
        numpy arrays of length ``2 * nc`` from Phase 1 preprocessing.
    device : torch.device

    Returns
    -------
    torch.Tensor
        Shape ``[batch, 2*nc]`` — z-scored conditioning vectors on ``device``.
    """
    patches_cpu = patches.detach().cpu()
    batch, nc, l, _ = patches_cpu.shape

    vf = patches_cpu.mean(dim=[2, 3])  # [batch, nc]

    ps_np = np.zeros((batch, nc), dtype=np.float32)
    patches_np = patches_cpu.numpy()
    for b in range(batch):
        for c in range(nc):
            mask = patches_np[b, c] > 0.5
            if mask.any():
                dt = scipy.ndimage.distance_transform_edt(mask)
                ps_np[b, c] = float(dt[mask].mean())

    ps = torch.from_numpy(ps_np)
    cond = torch.cat([vf, ps], dim=1).float()  # [batch, 2*nc]

    mean_t = torch.tensor(norm_stats[0], dtype=torch.float32)
    std_t = torch.tensor(norm_stats[1], dtype=torch.float32).clamp(min=1e-8)
    cond = (cond - mean_t) / std_t

    return cond.to(device)


def _conditional_training_loop(
    datasets,
    netG,
    netD,
    cond_enc,
    nc,
    l,
    nz,
    pth,
    norm_stats,
    isotropic=True,
    imtype='nphase',
    schedule=None,
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
    """Conditional WGAN-GP training loop.

    Extends ``_training_loop`` with:
    - ``cond_enc`` (ConditioningEncoder) has its own Adam optimizer.
    - Conditioning maps are appended to real and fake slices for the discriminator.
    - Regression loss (MSE on real slices) is added to the discriminator loss.
    - ``lambda_reg`` follows the staged curriculum.

    Parameters mirror ``_training_loop``; additions:

    cond_enc : ConditioningEncoder
        Maps ``[batch, 2*nc]`` conditioning vectors to ``[batch, embed_dim, lz, lz, lz]``.
    norm_stats : tuple of (mean_np, std_np)
        Arrays of length ``2*nc`` for normalising conditioning vectors.
    schedule : LambdaSchedule or None
        Curriculum schedule. Defaults to ``LambdaSchedule()``.

    Returns
    -------
    dict
        Same keys as ``_training_loop`` return dict.
    """
    if schedule is None:
        schedule = LambdaSchedule()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = netG.to(device)
    cond_enc = cond_enc.to(device)

    if isotropic:
        single_D = netD.to(device) if not isinstance(netD, (list, tuple)) else netD[0].to(device)
        netDs = [single_D, single_D, single_D]
        single_opt = Adam(single_D.parameters(), lr=lrd, betas=(beta1, beta2))
        optDs = [single_opt, single_opt, single_opt]
    else:
        netD_list = netD if isinstance(netD, (list, tuple)) else [netD, netD, netD]
        netDs = [d.to(device) for d in netD_list]
        optDs = [Adam(d.parameters(), lr=lrd, betas=(beta1, beta2)) for d in netDs]

    optG = Adam(
        list(netG.parameters()) + list(cond_enc.parameters()),
        lr=lrg, betas=(beta1, beta2),
    )

    loaderx = DataLoader(datasets[0], batch_size=D_batch_size, shuffle=True,
                         num_workers=workers, drop_last=True)
    loadery = DataLoader(datasets[1], batch_size=D_batch_size, shuffle=True,
                         num_workers=workers, drop_last=True)
    loaderz = DataLoader(datasets[2], batch_size=D_batch_size, shuffle=True,
                         num_workers=workers, drop_last=True)

    disc_real_log: list = []
    disc_fake_log: list = []
    gp_log: list = []
    Wass_log: list = []

    steps = min(len(loaderx), len(loadery), len(loaderz))
    start = time.time()
    fake_vol = None
    cond_dim = 2 * nc

    for epoch in range(num_epochs):
        lambda_reg = schedule.get(epoch + 1)

        for i, (bx, by, bz) in enumerate(zip(loaderx, loadery, loaderz)):
            real_batches = [bx[0].to(device), by[0].to(device), bz[0].to(device)]

            # ── Discriminator update ─────────────────────────────────────────
            with torch.no_grad():
                z = torch.randn(batch_size, nz, lz, lz, lz, device=device)
                # Sample conditioning from the first axis real batch
                cond_real = sample_conditioning_from_batch(real_batches[0], norm_stats, device)
                # Tile to match the number of fake slices
                n_fake = batch_size * l
                cond_for_fake = cond_real[:1].expand(n_fake, -1)
                emb = cond_enc(cond_real[:1].expand(batch_size, -1))
                gen_input = torch.cat([z, emb], dim=1)
                fake_vol = netG(gen_input)

            disc_real_acc = 0.0
            disc_fake_acc = 0.0
            gp_acc = 0.0

            for j in range(3):
                real_slice = real_batches[j]  # [D_batch, nc, l, l]
                fake_slice = (
                    fake_vol
                    .permute(0, _D1[j], 1, _D2[j], _D3[j])
                    .reshape(n_fake, nc, l, l)
                )

                # Tile real to match fake count
                tile = (n_fake + D_batch_size - 1) // D_batch_size
                real_tiled = real_slice.repeat(tile, 1, 1, 1)[:n_fake]

                # Conditioning maps for real and fake
                cond_maps = prepare_conditioning_maps_2d(cond_for_fake, l)  # [n_fake, cond_dim, l, l]
                real_cond = torch.cat([real_tiled, cond_maps], dim=1)
                fake_cond = torch.cat([fake_slice.detach(), cond_maps], dim=1)

                netDs[j].zero_grad()
                out_real, reg_real = netDs[j](real_cond)
                out_fake, _ = netDs[j](fake_cond)

                # Regression loss on real slices vs normalised conditioning
                cond_target = cond_for_fake  # already normalised
                reg_loss = F.mse_loss(reg_real, cond_target)

                gp = calc_gradient_penalty(
                    netDs[j], real_cond, fake_cond,
                    n_fake, l, device, Lambda, nc + cond_dim,
                )
                disc_cost = out_fake.mean() - out_real.mean() + gp + lambda_reg * reg_loss
                disc_cost.backward()
                optDs[j].step()

                disc_real_acc += out_real.mean().item()
                disc_fake_acc += out_fake.mean().item()
                gp_acc += gp.item()

            disc_real_log.append(disc_real_acc / 3)
            disc_fake_log.append(disc_fake_acc / 3)
            gp_log.append(gp_acc / 3)
            Wass_log.append((disc_real_acc - disc_fake_acc) / 3)

            # ── Generator update ─────────────────────────────────────────────
            if i % critic_iters == 0:
                optG.zero_grad()
                z = torch.randn(batch_size, nz, lz, lz, lz, device=device)
                cond_g = cond_real[:1].expand(batch_size, -1)
                emb_g = cond_enc(cond_g)
                fake_vol_g = netG(torch.cat([z, emb_g], dim=1))
                cond_maps_g = prepare_conditioning_maps_2d(
                    cond_g[:1].expand(n_fake, -1), l
                )
                errG = sum(
                    -netDs[j](
                        torch.cat([
                            fake_vol_g
                            .permute(0, _D1[j], 1, _D2[j], _D3[j])
                            .reshape(n_fake, nc, l, l),
                            cond_maps_g,
                        ], dim=1)
                    )[0].mean()
                    for j in range(3)
                )
                errG.backward()
                optG.step()
                fake_vol = fake_vol_g.detach()

            # ── Checkpoint ───────────────────────────────────────────────────
            if i % 25 == 0 and pth:
                torch.save(netG, pth + '_Gen.pt')
                torch.save(netDs[0], pth + '_Disc.pt')
                torch.save(cond_enc, pth + '_CondEnc.pt')
                util.test_plotter(fake_vol.cpu(), 8, imtype, pth)
                util.calc_eta(steps, time.time(), start, i, epoch, num_epochs)

    return {
        'disc_real_log': disc_real_log,
        'disc_fake_log': disc_fake_log,
        'gp_log': gp_log,
        'Wass_log': Wass_log,
        'netDs': netDs,
    }


def train_conditional(
    pth, imtype, datatype, real_data, Disc, Gen, CondEnc,
    nc, l, nz, sf,
    norm_stats,
    embed_dim=32,
    conditioning_dim=None,
    schedule=None,
):
    """Public entry point for conditional training.

    Parameters
    ----------
    pth : str
        Checkpoint path prefix.
    imtype : str
        Image type (e.g. 'nphase').
    datatype : str
        Data type for preprocessing (e.g. 'mat3D').
    real_data : list of str
        File paths. Length 1 → isotropic.
    Disc : class
        Discriminator2D constructor.
    Gen : class
        Generator3D constructor.
    CondEnc : class
        ConditioningEncoder constructor.
    nc : int
        Number of phases.
    l : int
        Spatial patch size.
    nz : int
        Latent noise channels.
    sf : int
        Scale factor for preprocessing.
    norm_stats : tuple of (mean_np, std_np)
        Conditioning normalisation stats from Phase 1.
    embed_dim : int
        Channels of conditioning embedding concatenated with z.
    conditioning_dim : int or None
        Length of conditioning vector. Defaults to ``2 * nc``.
    schedule : LambdaSchedule or None
        Curriculum schedule. Defaults to ``LambdaSchedule()``.
    """
    if conditioning_dim is None:
        conditioning_dim = 2 * nc

    isotropic = len(real_data) == 1
    if isotropic:
        real_data = real_data * 3

    gf = list(_GF)
    gf[-1] = nc

    datasets = preprocessing.batch(real_data, datatype, l, sf)

    netG = Gen(nz=nz, n_phases=nc, gk=_GK, gs=_GS, gf=gf, gp=_GP, embed_dim=embed_dim)
    netD = Disc(n_phases=nc, n_conditions=conditioning_dim, dk=_DK, ds=_DS, df=_DF, dp=_DP)
    cond_enc = CondEnc(conditioning_dim=conditioning_dim, embed_dim=embed_dim, lz=_LZ)

    _conditional_training_loop(
        datasets, netG, netD, cond_enc, nc, l, nz, pth,
        norm_stats=norm_stats, isotropic=isotropic, imtype=imtype,
        schedule=schedule,
    )


def generate_conditioned_samples(
    netG, cond_enc, cond_vector, n_samples, nz, lz, norm_stats, device
):
    """Generate hard-segmented volumes conditioned on a target conditioning vector.

    Parameters
    ----------
    netG : Generator3D
        Trained generator (with embed_dim > 0).
    cond_enc : ConditioningEncoder
        Trained conditioning encoder.
    cond_vector : array-like
        Raw (unnormalised) conditioning vector of length ``2 * n_phases``.
    n_samples : int
        Number of volumes to generate.
    nz : int
        Latent noise channels.
    lz : int
        Latent spatial size.
    norm_stats : tuple of (mean_np, std_np)
        Normalisation stats used during training.
    device : torch.device or str

    Returns
    -------
    torch.Tensor
        Shape ``[n_samples, n_phases, l, l, l]`` — argmax-segmented (long).
    """
    device = torch.device(device) if isinstance(device, str) else device
    netG = netG.to(device).eval()
    cond_enc = cond_enc.to(device).eval()

    cond_np = np.asarray(cond_vector, dtype=np.float32)
    mean_t = torch.tensor(norm_stats[0], dtype=torch.float32, device=device)
    std_t = torch.tensor(norm_stats[1], dtype=torch.float32, device=device).clamp(min=1e-8)
    cond_norm = (torch.tensor(cond_np, device=device) - mean_t) / std_t  # [2*nc]
    cond_batch = cond_norm.unsqueeze(0).expand(n_samples, -1)  # [n_samples, 2*nc]

    with torch.no_grad():
        emb = cond_enc(cond_batch)  # [n_samples, embed_dim, lz, lz, lz]
        z = torch.randn(n_samples, nz, lz, lz, lz, device=device)
        gen_input = torch.cat([z, emb], dim=1)
        soft_out = netG(gen_input)  # [n_samples, n_phases, l, l, l]

    return soft_out


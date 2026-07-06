from pathlib import Path

from slicegan.model import _training_loop
from slicegan.networks import Discriminator2D, Generator3D
from slicegan.preprocessing import batch


DATA_PATHS = [Path("segmented_dataset/Cell_0A.mat")]
OUT_PREFIX = "output/unconditional"


def train(
    data_paths=DATA_PATHS,
    pth=OUT_PREFIX,
    imtype="mat3D",
    nc=3,
    l=64,
    nz=64,
    sf=1,
    epochs=1,
    batch_size=8,
    d_batch_size=8,
    critic_iters=5,
    lz=4,
    g_arch=([4, 4, 4, 4, 4], [2, 2, 2, 2, 2], [512, 256, 128, 64, 3], [2, 2, 2, 2, 3]),
    d_arch=([4, 4, 4, 4, 4], [2, 2, 2, 2, 2], [64, 128, 256, 512, 512], [1, 1, 1, 1, 0]),
):
    datasets = batch([str(p) for p in data_paths], imtype=imtype, l=l, sf=sf)
    gk, gs, gf, gp = g_arch
    dk, ds, df, dp = d_arch
    gf = list(gf)
    gf[-1] = nc

    netG = Generator3D(nz=nz, n_phases=nc, gk=gk, gs=gs, gf=gf, gp=gp)
    makeD = lambda: Discriminator2D(n_phases=nc, n_conditions=0, dk=dk, ds=ds, df=df, dp=dp)
    isotropic = len(data_paths) == 1
    netD = makeD() if isotropic else [makeD(), makeD(), makeD()]
    if pth:
        Path(pth).parent.mkdir(parents=True, exist_ok=True)

    return _training_loop(
        datasets, netG, netD, nc, l, nz, pth=pth, isotropic=isotropic,
        imtype=imtype, num_epochs=epochs, batch_size=batch_size,
        D_batch_size=d_batch_size, critic_iters=critic_iters, lz=lz,
    )


if __name__ == "__main__":
    train()

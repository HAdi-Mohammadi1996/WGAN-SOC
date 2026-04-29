"""
SliceGAN network architectures extended with conditioning.

Generator3D and Discriminator2D mirror the SliceGAN networks.py structure.
The discriminator adds a regression head for auxiliary conditioning prediction.
"""

import torch
import torch.nn as nn


class Generator3D(nn.Module):
    """3D conditional generator using transpose convolutions.

    Architecture follows SliceGAN Table 1. Each layer except the last uses
    BatchNorm3d + ReLU; the final layer uses Softmax(dim=1) for n-phase output.

    Parameters
    ----------
    nz : int
        Latent noise channels.
    n_phases : int
        Number of output phases (= output channels after Softmax).
    gk, gs, gf, gp : list of int
        Per-layer kernel sizes, strides, output filter counts, and padding.
    embed_dim : int
        Conditioning embedding channels concatenated to z before layer 1.
    """

    def __init__(self, nz, n_phases, gk, gs, gf, gp, embed_dim=0):
        super().__init__()
        in_ch = nz + embed_dim
        layers = []
        n_layers = len(gk)
        for i in range(n_layers):
            k, s, f, p = gk[i], gs[i], gf[i], gp[i]
            layers.append(nn.ConvTranspose3d(in_ch, f, k, s, p, bias=False))
            if i < n_layers - 1:
                layers.append(nn.BatchNorm3d(f))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Softmax(dim=1))
            in_ch = f
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator2D(nn.Module):
    """2D slice discriminator with Wasserstein critic and auxiliary regression heads.

    Input is a 2D slice optionally concatenated with spatially tiled conditioning
    channels. No BatchNorm — follows SliceGAN discriminator design.

    Parameters
    ----------
    n_phases : int
        Number of phase channels in input slices (also controls regression output).
    n_conditions : int
        Extra conditioning channels appended to input (0 = unconditional).
    dk, ds, df, dp : list of int
        Per-layer kernel sizes, strides, filter counts, and padding.
    """

    def __init__(self, n_phases, n_conditions, dk, ds, df, dp):
        super().__init__()
        in_ch = n_phases + n_conditions
        feat_layers = []
        for k, s, f, p in zip(dk, ds, df, dp):
            feat_layers.append(nn.Conv2d(in_ch, f, k, s, p, bias=False))
            feat_layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = f
        self.features = nn.Sequential(*feat_layers)
        feat_dim = df[-1]
        self.critic_head = nn.Linear(feat_dim, 1)
        self.regression_head = nn.Sequential(
            nn.Linear(feat_dim, 2 * n_phases),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.critic_head(h), self.regression_head(h)


def slicegan_nets(path, training, imtype, dk, ds, df, dp, gk, gs, gf, gp,
                  nz=64, n_phases=3, n_conditions=0, embed_dim=0):
    """Factory matching the SliceGAN networks.slicegan_nets signature.

    Parameters
    ----------
    path : str
        Checkpoint path prefix (used when training=False to load weights).
    training : bool
        If False, loads saved weights from path + '_Gen.pt'.
    imtype : str
        Image type string (passed through for compatibility, not used here).
    dk, ds, df, dp : list
        Discriminator per-layer kernel, stride, filter, padding.
    gk, gs, gf, gp : list
        Generator per-layer kernel, stride, filter, padding.
    nz : int
        Latent noise channels.
    n_phases : int
        Number of output phases.
    n_conditions : int
        Extra conditioning input channels for discriminator.
    embed_dim : int
        Conditioning embedding dim for generator.

    Returns
    -------
    tuple of (Generator3D, Discriminator2D)
    """
    netG = Generator3D(
        nz=nz, n_phases=n_phases, gk=gk, gs=gs, gf=gf, gp=gp, embed_dim=embed_dim
    )
    netD = Discriminator2D(
        n_phases=n_phases, n_conditions=n_conditions, dk=dk, ds=ds, df=df, dp=dp
    )
    if not training:
        netG.load_state_dict(torch.load(path + '_Gen.pt', map_location='cpu'))
    return netG, netD

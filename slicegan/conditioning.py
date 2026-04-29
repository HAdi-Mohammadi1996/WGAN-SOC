"""
Conditioning utilities for Conditional SliceGAN.

ConditioningEncoder maps a per-sample conditioning vector to 3D spatial feature
maps that are concatenated with z before the generator's first layer.
prepare_conditioning_maps_2d tiles a vector to match 2D discriminator input.
"""

import torch
import torch.nn as nn


class ConditioningEncoder(nn.Module):
    """Maps conditioning vector to 3D spatial feature maps.

    Three linear layers with ReLU; output reshaped to [batch, embed_dim, lz, lz, lz]
    for concatenation with the latent noise tensor z.

    Parameters
    ----------
    conditioning_dim : int
        Length of input conditioning vector (e.g. 2 * n_phases).
    embed_dim : int
        Number of output feature map channels.
    lz : int
        Spatial size of output feature maps (same as generator input spatial size).
    hidden_dim : int
        Width of the hidden linear layers.
    """

    def __init__(self, conditioning_dim, embed_dim, lz=4, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.lz = lz
        out_dim = embed_dim * lz * lz * lz
        self.net = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond):
        """
        Parameters
        ----------
        cond : torch.Tensor
            Shape [batch, conditioning_dim].

        Returns
        -------
        torch.Tensor
            Shape [batch, embed_dim, lz, lz, lz].
        """
        out = self.net(cond)
        return out.view(cond.size(0), self.embed_dim, self.lz, self.lz, self.lz)


def prepare_conditioning_maps_2d(cond_vector, spatial_size):
    """Tile a conditioning vector to 2D spatial maps for discriminator input.

    Parameters
    ----------
    cond_vector : torch.Tensor
        Shape [batch, conditioning_dim].
    spatial_size : int
        Target spatial height and width.

    Returns
    -------
    torch.Tensor
        Shape [batch, conditioning_dim, spatial_size, spatial_size].
        Uses expand (no copy) — call .contiguous() if needed.
    """
    batch, c = cond_vector.shape
    return cond_vector.view(batch, c, 1, 1).expand(batch, c, spatial_size, spatial_size)

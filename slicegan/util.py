import time
import torch
import torch.autograd as autograd
import torchvision.utils


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, Lambda, nc):
    """WGAN-GP gradient penalty.

    Parameters
    ----------
    netD : nn.Module
        Discriminator whose forward() returns (critic_score, regression_pred).
    real_data, fake_data : Tensor
        Both shape [batch_size, nc, l, l].
    batch_size : int
        Number of samples (= l * D_batch_size in training).
    l : int
        Spatial size of 2D slices.
    device : torch.device
    Lambda : float
        GP coefficient.
    nc : int
        Number of input channels.

    Returns
    -------
    Tensor
        Scalar gradient penalty.
    """
    alpha = torch.rand(batch_size, 1, l, l, device=device).expand(batch_size, nc, l, l)
    interpolated = (
        alpha * real_data.detach() + (1.0 - alpha) * fake_data.detach()
    ).requires_grad_(True)

    disc_score, _ = netD(interpolated)

    gradients = autograd.grad(
        outputs=disc_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return Lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def test_plotter(img, n_slices, imtype, path):
    """Save n_slices cross-sections from the first volume in img as a PNG.

    Parameters
    ----------
    img : Tensor
        Shape [batch, nc, l, l, l].
    n_slices : int
        Number of evenly-spaced slices to save.
    imtype : str
        Passed through for API compatibility (unused here).
    path : str
        Path prefix; saves to ``{path}_slices.png``. Skipped if falsy.
    """
    if not path:
        return
    vol = img[0].detach().cpu()   # [nc, l, l, l]
    l = vol.shape[-1]
    nc = vol.shape[0]
    label = vol.argmax(dim=0).float()   # [l, l, l]  values in [0, nc-1]
    if nc > 1:
        label = label / (nc - 1)
    indices = torch.linspace(0, l - 1, n_slices).long()
    slices = label[indices, :, :].unsqueeze(1)   # [n_slices, 1, l, l]
    torchvision.utils.save_image(slices, f"{path}_slices.png", nrow=n_slices)


def calc_eta(steps, time_now, start, i, epoch, num_epochs):
    """Print estimated time remaining.

    Parameters
    ----------
    steps : int
        Number of inner-loop steps per epoch.
    time_now : float
        Current timestamp from time.time().
    start : float
        Training start timestamp.
    i : int
        Current inner-loop step index (0-based).
    epoch : int
        Current epoch index (0-based).
    num_epochs : int
        Total number of epochs.
    """
    elapsed = time_now - start
    total = num_epochs * steps
    done = epoch * steps + i + 1
    remaining = elapsed / max(done, 1) * max(total - done, 0)
    h = int(remaining // 3600)
    m = int((remaining % 3600) // 60)
    s = int(remaining % 60)
    print(f"[Epoch {epoch + 1}/{num_epochs} | Step {i + 1}/{steps}] ETA: {h:02d}:{m:02d}:{s:02d}")

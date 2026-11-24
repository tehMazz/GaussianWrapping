"""Visibility ledger (H1): seen_i = sum over train views and pixels of alpha_i * T_i."""
import torch

from gaussian_renderer.beags_api import contribution


@torch.no_grad()
def compute_seen(cameras, gaussians, pipe):
    """(P,) total pixel weight each Gaussian contributes to the training images."""
    seen = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
    for cam in cameras:
        seen += contribution(cam, gaussians, pipe)
    return seen


def prune(gaussians, mask):
    """gaussians.prune_points(mask) (syncs extras and filter_3D) unless the mask is empty; returns the count."""
    n = int(mask.sum())
    if n:
        gaussians.prune_points(mask)
    return n


def prune_unseen(gaussians, seen, tau=1.0):
    """Prune seen < tau; returns the count.

    seen is in pixels: tau = 1 removes Gaussians that supply less than one pixel of colour to the
    whole training set. Removing Gaussian i changes a pixel by alpha_i T_i (c_i - C_behind), so the
    summed absolute change of all training renders is at most seen_i < tau (colours in [0, 1]):
    the photometric loss can neither see nor constrain them. The bound is per Gaussian, so tau
    does not depend on scene size or view count.
    """
    return prune(gaussians, seen < tau)

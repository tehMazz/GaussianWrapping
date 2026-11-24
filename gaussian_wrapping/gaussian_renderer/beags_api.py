"""BEA-GS rendering helpers on top of render_ours (labels, depth/alpha, per-Gaussian contribution)."""
import torch

from gaussian_renderer.ours import render_ours


def render_labels(cam, gaussians, pipe, bg, override_color=None, mask=None, gaussian_labels=None,
                  pixel_labels=None, num_labels=0):
    """render_ours dict; with labels it also holds "sag" (1, H, W) and "votes" (P, L)."""
    return render_ours(cam, gaussians, pipe, bg, colors_precomp=override_color, mask=mask,
                       gaussian_labels=gaussian_labels, pixel_labels=pixel_labels, num_labels=num_labels)


@torch.no_grad()
def render_depth_alpha(cam, gaussians, pipe, mask=None):
    """Median z-depth [H, W] (0 where alpha < 0.5 or no surface) and alpha [H, W]."""
    bg = torch.zeros(3, device="cuda")
    out = render_ours(cam, gaussians, pipe, bg, require_depth=True, mask=mask)
    depth, alpha = out["median_depth"][0], out["alpha"][0]
    return torch.where(alpha < 0.5, torch.zeros_like(depth), depth), alpha


@torch.no_grad()
def contribution(cam, gaussians, pipe, mask=None):
    """[P] sum over pixels of alpha*T (votes with L=1, every pixel labelled 0)."""
    n = gaussians.get_xyz.shape[0]
    bg = torch.zeros(3, device="cuda")
    gaussian_labels = torch.zeros(n, dtype=torch.int32, device="cuda")
    pixel_labels = torch.zeros((int(cam.image_height), int(cam.image_width)), dtype=torch.int32, device="cuda")
    out = render_ours(cam, gaussians, pipe, bg, require_depth=False, mask=mask, gaussian_labels=gaussian_labels,
                      pixel_labels=pixel_labels, num_labels=1)
    return out["votes"][:, 0]


@torch.no_grad()
def transmittance(cam, gaussians, pipe, mask=None):
    """[P, 3] GAVIS single-view terms: sum of T in front of each Gaussian over the pixels it covers, their count,
    and its footprint area in pixels (render_ours vis=True)."""
    bg = torch.zeros(3, device="cuda")
    out = render_ours(cam, gaussians, pipe, bg, require_depth=False, mask=mask, vis=True)
    return out["vis"]

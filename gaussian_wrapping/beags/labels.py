"""Per-Gaussian labels: voting from 2D masks, label renders, labelled point cloud."""
import torch

from beags.consensus import backproject
from gaussian_renderer.beags_api import render_depth_alpha, render_labels


def labels_from_votes(votes):
    """(P, L) votes -> (P,) int32 argmax; -1 for Gaussians without any vote (never hit)."""
    return torch.where(votes.sum(dim=1) > 0, votes.argmax(dim=1), -1).int()


@torch.no_grad()
def assign_labels(gaussians, cameras, pipe, num_labels):
    """label = argmax_l sum_views sum_pixels(label l) alpha*T. Never-hit Gaussians get -1."""
    n = gaussians.get_xyz.shape[0]
    votes = torch.zeros((n, num_labels), device="cuda")
    dummy = torch.zeros(n, dtype=torch.int32, device="cuda")  # labels only switch voting on here
    bg = torch.zeros(3, device="cuda")
    for cam in cameras:
        if (cam.masks.labels >= 0).any():
            votes += render_labels(cam, gaussians, pipe, bg, gaussian_labels=dummy,
                                   pixel_labels=cam.masks.labels, num_labels=num_labels)["votes"]
    return labels_from_votes(votes)


@torch.no_grad()
def label_coverage(cam, gaussians, pipe, labels, label, extracted=False):
    """(H, W) weight of `label` Gaussians: rendered with all Gaussians (occluded by others),
    or alone (`extracted=True`, the object cut out of the scene)."""
    ones = torch.ones_like(gaussians.get_xyz)
    colors = ones * (labels == label)[:, None]
    mask = (labels == label) if extracted else None
    bg = torch.zeros(3, device="cuda")
    return render_labels(cam, gaussians, pipe, bg, override_color=colors, mask=mask)["render"].mean(dim=0)


@torch.no_grad()
def labeled_point_cloud(gaussians, cameras, pipe, labels, threshold=0.5, label_ids=None,
                        output_device="cuda"):
    """Back-project every pixel where a label covers >= threshold. Returns (xyz, label).
    label_ids: labels to back-project; default all object labels (1..max), i.e. no background."""
    if label_ids is None:
        label_ids = range(1, int(labels.max()) + 1)
    xyz, lab = [], []
    for cam in cameras:
        depth, _ = render_depth_alpha(cam, gaussians, pipe)
        world = backproject(depth, cam)
        for label in label_ids:
            hit = (label_coverage(cam, gaussians, pipe, labels, label) >= threshold) & (depth > 0)
            xyz.append(world[hit].to(output_device))
            lab.append(torch.full((int(hit.sum()),), label, dtype=torch.int32, device=output_device))
    return torch.cat(xyz), torch.cat(lab)

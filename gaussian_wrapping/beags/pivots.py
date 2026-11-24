"""Pivot term: the front pivot of a labelled Gaussian must lie near voxels of its own label.

L_piv = mean_i o'_i * relu(d_{l_i}(p_i) - h) / h,  p_i = mu_i + 3 ||S R^T n_i|| n_i (GW front pivot),
h = voxel edge. d is the Chebyshev ring of the first same-label voxel times h (0 inside the 27-neighbourhood),
or, with a MeshField, h = the label's band and d = distance to the label's mesh region (0 inside);
the gradient only reaches the opacity.
"""
import torch

from regularization.normal_field import get_gaussian_std_in_direction


@torch.no_grad()
def front_pivots(gaussians, idx, std_factor=3.0):
    """(len(idx), 3) mu + std_factor * ||S R^T n|| n, n the learned normal (smallest axis without normal field)."""
    if gaussians.use_gaussian_features:
        normals = gaussians.convert_features_to_normals(normalize=True)[idx]
    else:
        normals = gaussians.get_smallest_axis[idx]
    normals = torch.nn.functional.normalize(normals, dim=-1)
    stds = get_gaussian_std_in_direction(
        directions=normals[:, None], gaussian_scaling=gaussians.get_scaling_with_3D_filter[idx],
        gaussian_rotation=gaussians._rotation[idx], normalize_directions=False)        # (N, 1)
    return gaussians.get_xyz[idx] + std_factor * stds * normals


def label_distance(voxels, points, labels, max_ring=3, chunk=2_000_000):
    """(N,) distance of each point to its label's region: LabeledVoxels rings or MeshField (beags/mesh_field.py)."""
    return voxels.distance(points, labels, max_ring, chunk)


def pivot_loss(gaussians, labels, voxels, max_ring=3, max_gaussians=200_000):
    """L_piv over Gaussians with label >= 0 (a random subset of max_gaussians per call; None: all)."""
    idx = (labels >= 0).nonzero().squeeze(1)
    if not idx.numel():
        return gaussians.get_xyz.sum() * 0
    if max_gaussians is not None and idx.numel() > max_gaussians:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_gaussians]]
    h = voxels.tolerance(labels[idx])
    d = label_distance(voxels, front_pivots(gaussians, idx), labels[idx], max_ring)
    return (gaussians.get_opacity_with_3D_filter[idx, 0] * torch.relu(d - h) / h).mean()

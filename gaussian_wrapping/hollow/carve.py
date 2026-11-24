"""See-through carving (H4): Gaussians whose centre the training views see through.

Vacancy is GW's: transmittance at a 3D point from the SOF ray evaluation (gaussian_renderer/sof.py
compute_transmittance, as evaluate_vacancy_sof). At a Gaussian's own centre the ray through it
peaks at the centre, so the Gaussian itself contributes exactly min(0.99, o') and
T(centre) <= 1 - o'. Excluding the Gaussian (T / (1 - o')) would flag every front-surface
Gaussian, so it is kept in: T > threshold then means o' < 1 - threshold AND nothing opaque in
front, i.e. faint Gaussians floating unobstructed in observed space (see-through fog). Opaque
Gaussians are never carved. Requiring this in >= 2 views and seen > 0 drops single-view
grazing hits and Gaussians that no training pixel uses.
"""
import torch

from gaussian_renderer.sof import GlobalSortOrder, compute_transmittance, default_splat_args, get_frustum_mask
from hollow.ledger import compute_seen
from utils.camera_utils import get_cameras_spatial_extent


@torch.no_grad()
def carve_see_through(gaussians, cameras, pipe, threshold=0.9, min_views=2, seen=None):
    """(P,) bool: T(centre) > threshold in >= min_views training views and seen > 0."""
    if seen is None:
        seen = compute_seen(cameras, gaussians, pipe)
    splat_args = default_splat_args()
    splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
    splat_args.meshing_settings.alpha_early_stop = True  # exact for the T > threshold test
    splat_args.meshing_settings.transmittance_threshold = threshold
    radius = get_cameras_spatial_extent(cameras=cameras)["radius"].item()
    near, far = 0.02 * radius / 6.0, 1e6 * radius / 6.0  # as regularization/mesh_in_the_loop.py

    points = gaussians.get_xyz
    background = torch.zeros(3, device=points.device)
    views = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
    for cam in cameras:
        frustum = get_frustum_mask(points=points, cameras=[cam], near=near, far=far)
        if not frustum.any():
            continue
        T = compute_transmittance(points3D=points[frustum], viewpoint_camera=cam, pc=gaussians, pipe=pipe,
                                  bg_color=background, kernel_size=0.0, splat_args=splat_args)["transmittance"]
        views[frustum] += (T > threshold).int()
    return (views >= min_views) & (seen > 0)

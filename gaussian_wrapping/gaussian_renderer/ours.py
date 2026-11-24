from typing import Optional
import math
import torch
from diff_gaussian_rasterization_ours import GaussianRasterizationSettings, GaussianRasterizer
try:
    # BEA-GS build of the same rasterizer (adds labels: SAG + votes); identical output without labels
    from diff_gaussian_rasterization_ours_beags import GaussianRasterizationSettings as LabelRasterizationSettings
    from diff_gaussian_rasterization_ours_beags import GaussianRasterizer as LabelRasterizer
except ImportError:
    LabelRasterizationSettings, LabelRasterizer = None, None
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


def render_ours(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    kernel_size = 0.0, 
    scaling_modifier = 1.0, 
    require_coord : bool = False,
    require_depth : bool = True,
    colors_precomp : Optional[torch.Tensor]=None,
    mask : Optional[torch.Tensor]=None,
    gaussian_labels : Optional[torch.Tensor]=None,
    pixel_labels : Optional[torch.Tensor]=None,
    num_labels : int=0,
    vis : bool=False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    mask: bool [P], only the selected Gaussians are rasterized. radii, visibility_filter and votes
        are scattered back to full length P; viewspace_points stays full length.
    gaussian_labels: int [P], pixel_labels: int [H, W] (-1 = unlabeled), num_labels: L.
        Adds "sag" (1, H, W), the sum of alpha*T over Gaussians whose label != pixel label
        (gradient only to those Gaussians), and "votes" (P, L), the sum of alpha*T over pixels
        with label l (no grad). Both are None without labels. Requires diff_gaussian_rasterization_ours_beags.
    vis: adds "vis" (P, 5), no grad (GAVIS visibility, beags/visibility.py): per Gaussian the sum of the
        transmittance T in front of it over the pixels it covers, the number of those pixels, its footprint
        area in pixels (0 when culled), the max of alpha*T over the labelled pixels it covers (0 without
        pixel_labels: "does it dominate some pixel"), and the sum of alpha*T over every pixel it covers
        (its blend mass, gs-atoms EpochPrune). None when off. Requires diff_gaussian_rasterization_ours_beags.
    """
    use_labels = gaussian_labels is not None
    if LabelRasterizer is not None:
        RasterizationSettings, Rasterizer = LabelRasterizationSettings, LabelRasterizer
    elif use_labels or vis:
        raise ImportError("Rendering with labels or vis requires diff_gaussian_rasterization_ours_beags")
    else:
        RasterizationSettings, Rasterizer = GaussianRasterizationSettings, GaussianRasterizer

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    active_sg_degree = 0

    raster_settings = RasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_depth = require_depth,
        debug=pipe.debug
    )

    rasterizer = Rasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points  # full length P; a masked view is passed to the rasterizer

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    # TODO: Add 3D filter
    if pc.use_unbounded_opacity:
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    else:
        scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features if colors_precomp is None else None
    
    # sg_axis = pc.get_sg_axis
    # sg_sharpness = pc.get_sg_sharpness
    # sg_color = pc.get_sg_color
    
    n_total = pc._xyz.shape[0]
    if mask is not None:
        means3D, means2D, opacity, scales, rotations = means3D[mask], means2D[mask], opacity[mask], scales[mask], rotations[mask]
        shs = shs[mask] if shs is not None else None
        colors_precomp = colors_precomp[mask] if colors_precomp is not None else None
        gaussian_labels = gaussian_labels[mask] if use_labels else None

    n_gaussians = means3D.shape[0]
    sg_axis = torch.zeros(n_gaussians, active_sg_degree, 3, device=pc._xyz.device)
    sg_sharpness = torch.zeros(n_gaussians, active_sg_degree, device=pc._xyz.device)
    sg_color = torch.zeros(n_gaussians, active_sg_degree, 3, device=pc._xyz.device)

    label_kwargs = {}
    if use_labels:
        label_kwargs = dict(gaussian_labels=gaussian_labels, pixel_labels=pixel_labels, num_labels=num_labels)
    vis_out = None
    if vis:
        vis_out = torch.zeros(n_gaussians, 5, dtype=torch.float32, device=pc._xyz.device)
        label_kwargs["vis"] = vis_out
    outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sg_axis = sg_axis,
        sg_sharpness = sg_sharpness,
        sg_color = sg_color,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        **label_kwargs,
    )
    rendered_image, radii, rendered_median_depth, rendered_alpha, rendered_normal = outputs[:5]
    sag, votes = outputs[5:] if use_labels else (None, None)

    if mask is not None:
        radii_full = radii.new_zeros(n_total)
        radii_full[mask] = radii
        radii = radii_full
        if votes is not None:
            votes_full = votes.new_zeros(n_total, votes.shape[1])
            votes_full[mask] = votes
            votes = votes_full
        if vis_out is not None:
            vis_full = vis_out.new_zeros(n_total, 5)
            vis_full[mask] = vis_out
            vis_out = vis_full

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": None,
            "expected_coord": None,
            "median_coord": None,
            "expected_depth": rendered_median_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "alpha": rendered_alpha,
            "sag": sag,
            "votes": votes,
            "vis": vis_out,
            }


# integration is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py
def integrate_ours(points3D, viewpoint_camera, pc : GaussianModel, pipe, kernel_size : float, scaling_modifier = 1.0):
    """
    integrate Gaussians to the points
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    active_sg_degree = 0
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=None,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    depth_plane_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    alpha_integrated, inside = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"alpha_integrated": alpha_integrated,
            "inside": inside}
    
    
def sample_depth_with_ours(points3D, viewpoint_camera: Camera, pc : GaussianModel, pipe : torch.Tensor, kernel_size : float, scaling_modifier = 1.0):

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    active_sg_degree = 0

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth, inside = rasterizer.sample_depth(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"sampled_depth": depth,
            "inside": inside}
    
    
@torch.no_grad()
def sample_depth_and_normal_in_view(
    points: torch.Tensor, 
    viewpoint_camera: Camera, 
    gaussians: GaussianModel, 
    pipe, 
    background: torch.Tensor, 
    kernel_size: float=0.0, 
    scaling_modifier: float=1.0
):
    from utils.geometry_utils import sample_depth_normal, depth_to_normal_with_mask
    
    # Get depth and normal maps
    render_pkg = render_ours(
        viewpoint_camera=viewpoint_camera,
        pc=gaussians,
        pipe=pipe,
        bg_color=background,
        kernel_size=kernel_size,
        scaling_modifier=scaling_modifier,
    )
    depth = render_pkg["median_depth"]  # (1, H, W)
    normal, _ = depth_to_normal_with_mask(viewpoint_camera, depth)  # (3, H, W), (1, H, W)
    
    # Sample depth and normal for each point
    return sample_depth_normal(viewpoint_camera, depth, normal, points)
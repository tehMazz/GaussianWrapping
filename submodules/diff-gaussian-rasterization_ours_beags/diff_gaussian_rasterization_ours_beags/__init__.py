#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    sg_axis,
    sg_sharpness,
    sg_color,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    gaussian_labels=None,
    pixel_labels=None,
    num_labels=0,
    vis=None,
):
    # BEA-GS labels (None = stock rasterizer)
    if gaussian_labels is None:
        gaussian_labels = torch.empty(0, dtype=torch.int32, device=means3D.device)
        pixel_labels = gaussian_labels
    gaussian_labels = gaussian_labels.int().contiguous()
    pixel_labels = pixel_labels.reshape(-1).int().contiguous()
    if vis is None:
        vis = torch.empty(0, dtype=torch.float32, device=means3D.device)
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        gaussian_labels,
        pixel_labels,
        num_labels,
        vis,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        gaussian_labels,
        pixel_labels,
        num_labels,
        vis,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            sh,
            sg_axis,
            sg_sharpness,
            sg_color,
            raster_settings.sh_degree,
            raster_settings.sg_degree,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.require_depth,
            raster_settings.debug,
            gaussian_labels,
            pixel_labels,
            num_labels,
        )
        if vis.numel() > 0:  # only then: an extension built before vis takes the call without it
            args = args + (vis,)

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, alpha, normal, mdepth, radii, geomBuffer, binningBuffer, imgBuffer, tileBuffer, sag, votes = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, alpha, normal, mdepth, radii, geomBuffer, binningBuffer, imgBuffer, tileBuffer, sag, votes = _C.rasterize_gaussians(*args)
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_labels = num_labels
        ctx.save_for_backward(
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            sh,
            sg_axis,
            sg_sharpness,
            sg_color,
            alpha,
            normal,
            mdepth,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            tileBuffer,
            gaussian_labels,
            pixel_labels,
            sag,
        )
        ctx.mark_non_differentiable(votes)
        return color, radii, mdepth, alpha, normal, sag, votes

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_mdepth, grad_alpha, grad_normal, grad_sag, grad_votes):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            sh,
            sg_axis,
            sg_sharpness,
            sg_color,
            alpha,
            normal,
            mdepth,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            tileBuffer,
            gaussian_labels,
            pixel_labels,
            sag,
        ) = ctx.saved_tensors
        grad_sag = torch.zeros_like(sag) if grad_sag is None else grad_sag.contiguous()

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            sh,
            sg_axis,
            sg_sharpness,
            sg_color,
            raster_settings.sh_degree,
            raster_settings.sg_degree,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            grad_color,
            grad_mdepth,
            grad_alpha,
            grad_normal,
            alpha,
            normal,
            mdepth,
            raster_settings.campos,
            radii,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            tileBuffer,
            raster_settings.require_depth,
            raster_settings.debug,
            gaussian_labels,
            pixel_labels,
            ctx.num_labels,
            grad_sag,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_sg_axis,
                    grad_sg_sharpness,
                    grad_sg_color,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_sg_axis,
                grad_sg_sharpness,
                grad_sg_color,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_sg_axis,
            grad_sg_sharpness,
            grad_sg_color,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
            None,
            None,
        )
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    kernel_size: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    sg_degree: int
    campos: torch.Tensor
    prefiltered: bool
    require_depth: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(positions, raster_settings.viewmatrix, raster_settings.projmatrix)

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        sg_axis=None,
        sg_sharpness=None,
        sg_color=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        gaussian_labels=None,
        pixel_labels=None,
        num_labels=0,
        vis=None,
    ):
        """Returns color, radii, mdepth, alpha, normal, sag, votes.
        BEA-GS labels: gaussian_labels int [P], pixel_labels int [H, W] (-1 = unlabeled), num_labels L.
        sag (1, H, W): sum of alpha * T over Gaussians whose label != pixel label (differentiable).
        votes (P, L): sum of alpha * T over pixels with label l (no grad). Both empty without labels.
        vis: optional (P, 3) float32 CUDA zeros, filled in place (no grad; GAVIS visibility, independent of labels):
        sum of T in front of each Gaussian over the pixels it covers, the pixel count, the footprint area in pixels.
        """

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            sg_axis,
            sg_sharpness,
            sg_color,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            gaussian_labels,
            pixel_labels,
            num_labels,
            vis,
        )

    def integrate(
        self,
        points3D,
        means3D,
        opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        view2gaussian_precomp=None,
    ):

        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # TODO check and raise exception for precomputed view2gaussian
        if view2gaussian_precomp is None:
            view2gaussian_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        # Restructure arguments the way that the C++ lib expects them
        args = (
            points3D,
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            view2gaussian_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            0.0,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, transmittance, inside = _C.integrate_gaussians_to_points(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, transmittance, inside = _C.integrate_gaussians_to_points(*args)

        return 1 - transmittance, inside

    def evaluate_sdf(
        self,
        points3D,
        means3D,
        opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        view2gaussian_precomp=None,
    ):

        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # TODO check and raise exception for precomputed view2gaussian
        if view2gaussian_precomp is None:
            view2gaussian_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        # Restructure arguments the way that the C++ lib expects them
        args = (
            points3D,
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            view2gaussian_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            0.0,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, depth, sdf, inside = _C.evaluate_sdf_from_signle_view(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, depth, sdf, inside = _C.evaluate_sdf_from_signle_view(*args)

        return depth, sdf, inside

    def sample_depth(self, points3D, means3D, opacities, scales=None, rotations=None, cov3D_precomp=None):
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        raster_settings = self.raster_settings
        return _SampleDepth.apply(points3D, means3D, opacities, scales, rotations, cov3D_precomp, raster_settings)


class _SampleDepth(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points3D,
        means3D,
        opacities,
        scales,
        rotations,
        cov3D_precomp,
        raster_settings,
    ):

        # Invoke C++/CUDA rasterization routine
        # Restructure arguments the way that the C++ lib expects them
        args = (
            points3D,
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            0.0,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    num_points,
                    num_duplicated_tiles,
                    camera_points,
                    inside,
                    geomBuffer,
                    binningBuffer,
                    pointBuffer,
                    pointBinningBuffer,
                    tileBuffer,
                    duplicatedTileBuffer,
                ) = _C.sample_rasterized_depth(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            (
                num_rendered,
                num_points,
                num_duplicated_tiles,
                camera_points,
                inside,
                geomBuffer,
                binningBuffer,
                pointBuffer,
                pointBinningBuffer,
                tileBuffer,
                duplicatedTileBuffer,
            ) = _C.sample_rasterized_depth(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_points = num_points
        ctx.num_duplicated_tiles = num_duplicated_tiles
        ctx.save_for_backward(
            points3D,
            means3D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            inside,
            geomBuffer,
            binningBuffer,
            pointBuffer,
            pointBinningBuffer,
            tileBuffer,
            duplicatedTileBuffer,
        )
        return camera_points, inside

    @staticmethod
    def backward(ctx, grad_camera_points, grad_inside):
        num_rendered = ctx.num_rendered
        num_points = ctx.num_points
        num_duplicated_tiles = ctx.num_duplicated_tiles
        raster_settings = ctx.raster_settings
        (
            points3D,
            means3D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            inside,
            geomBuffer,
            binningBuffer,
            pointBuffer,
            pointBinningBuffer,
            tileBuffer,
            duplicatedTileBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            points3D,
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            inside,
            grad_camera_points,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            geomBuffer,
            binningBuffer,
            pointBuffer,
            pointBinningBuffer,
            tileBuffer,
            duplicatedTileBuffer,
            num_rendered,
            num_points,
            num_duplicated_tiles,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_opacities, grad_means3D, grad_cov3D_precomp, grad_scales, grad_rotations, grad_points3D = _C.sample_rasterized_depth_backward(
                    *args
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_opacities, grad_means3D, grad_cov3D_precomp, grad_scales, grad_rotations, grad_points3D = _C.sample_rasterized_depth_backward(*args)

        grads = (
            grad_points3D,
            grad_means3D,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3D_precomp,
            None,
        )
        return grads

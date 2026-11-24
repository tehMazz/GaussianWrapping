/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  // without the GAVIS vis buffer (the Python wrapper passes it only when requested)
  m.def("rasterize_gaussians",
        [](const torch::Tensor& background, const torch::Tensor& means3D, const torch::Tensor& colors,
           const torch::Tensor& opacity, const torch::Tensor& scales, const torch::Tensor& rotations,
           const torch::Tensor& cov3D_precomp, const torch::Tensor& sh, const torch::Tensor& sg_axis,
           const torch::Tensor& sg_sharpness, const torch::Tensor& sg_color, const int sh_degree, const int sg_degree,
           const float scale_modifier, const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
           const float tan_fovx, const float tan_fovy, const float kernel_size, const int image_height,
           const int image_width, const torch::Tensor& campos, const bool prefiltered, const bool require_depth,
           const bool debug, const torch::Tensor& gaussian_labels, const torch::Tensor& pixel_labels,
           const int num_labels) {
            return RasterizeGaussiansCUDA(background, means3D, colors, opacity, scales, rotations, cov3D_precomp, sh,
                                          sg_axis, sg_sharpness, sg_color, sh_degree, sg_degree, scale_modifier,
                                          viewmatrix, projmatrix, tan_fovx, tan_fovy, kernel_size, image_height,
                                          image_width, campos, prefiltered, require_depth, debug, gaussian_labels,
                                          pixel_labels, num_labels, torch::empty({0}, means3D.options()));
        });
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("integrate_gaussians_to_points", &IntegrateGaussiansToPointsCUDA);
  m.def("evaluate_sdf_from_signle_view", &evaluateSDFfromSingleView);
  m.def("mark_visible", &markVisible);
  m.def("sample_rasterized_depth", &SampleRasterizedDepthCUDA);
  m.def("sample_rasterized_depth_backward", &SampleRasterizedDepthBackwardCUDA);
}
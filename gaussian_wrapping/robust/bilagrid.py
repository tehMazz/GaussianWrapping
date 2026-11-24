# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 Yuehao Wang (https://github.com/yuehaowang). This part of code is borrowed form ["Bilateral Guided Radiance Field Processing"](https://bilarfpro.github.io/).
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-view 3D bilateral grids (gsplat examples/lib_bilagrid.py, without the CP-4D grid and tensorly).

Applied to the training render before the photometric loss; test, eval and export use the raw render.
"""
import torch
import torch.nn.functional as F
from torch import nn


def color_affine_transform(affine_mats, rgb):
    """affine_mats (..., 3, 4), rgb (..., 3) -> (..., 3)."""
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


def _num_tensor_elems(t):
    return max(torch.prod(torch.tensor(t.size()[1:]).float()).item(), 1.0)


def total_variation_loss(x):
    """Total variation of a (B, C, ...) tensor over all dims after C."""
    batch_size = x.shape[0]
    tv = 0
    for i in range(2, len(x.shape)):
        n_res = x.shape[i]
        idx1 = torch.arange(1, n_res, device=x.device)
        idx2 = torch.arange(0, n_res - 1, device=x.device)
        x1 = x.index_select(i, idx1)
        x2 = x.index_select(i, idx2)
        count = _num_tensor_elems(x1)
        tv += torch.pow((x1 - x2), 2).sum() / count
    return tv / batch_size


def slice(bil_grids, xy, rgb, grid_idx):
    """Slice grids at xy (..., 2) in [0, 1] with gray guidance of rgb (..., 3); grid_idx (..., 1).

    Returns {"rgb": (..., 3), "rgb_affine_mats": (..., 3, 4)}.
    """
    sh_ = rgb.shape

    grid_idx_unique = torch.unique(grid_idx)
    if len(grid_idx_unique) == 1:
        grid_idx = grid_idx_unique
        xy = xy.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
    else:
        if len(grid_idx.shape) == 4:
            grid_idx = grid_idx[:, 0, 0, 0]
        elif len(grid_idx.shape) == 3:
            grid_idx = grid_idx[:, 0, 0]
        elif len(grid_idx.shape) == 2:
            grid_idx = grid_idx[:, 0]
        else:
            raise ValueError("The input to bilateral grid slicing is not supported yet.")

    affine_mats = bil_grids(xy, rgb, grid_idx)
    rgb = color_affine_transform(affine_mats, rgb)

    return {
        "rgb": rgb.reshape(*sh_),
        "rgb_affine_mats": affine_mats.reshape(*sh_[:-1], affine_mats.shape[-2], affine_mats.shape[-1]),
    }


class BilateralGrid(nn.Module):
    """num 3D bilateral grids of affine colour transforms, grids (N, 12, L, H, W), identity init."""

    def __init__(self, num, grid_X=16, grid_Y=16, grid_W=8):
        super(BilateralGrid, self).__init__()
        self.grid_width = grid_X
        self.grid_height = grid_Y
        self.grid_guidance = grid_W

        grid = self._init_identity_grid()
        self.grids = nn.Parameter(grid.tile(num, 1, 1, 1, 1))

        # BT601 RGB-to-gray, guidance in [-1, 1].
        self.register_buffer("rgb2gray_weight", torch.Tensor([[0.299, 0.587, 0.114]]))
        self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2.0 - 1.0

    def _init_identity_grid(self):
        grid = torch.tensor([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]).float()
        grid = grid.repeat([self.grid_guidance * self.grid_height * self.grid_width, 1])
        grid = grid.reshape(1, self.grid_guidance, self.grid_height, self.grid_width, -1)
        return grid.permute(0, 4, 1, 2, 3)

    def tv_loss(self):
        return total_variation_loss(self.grids)

    def forward(self, grid_xy, rgb, idx=None):
        """2-5D grid_xy (..., 2) and rgb (..., 3) -> affine matrices (..., 3, 4); see `slice`."""
        input_ndims = len(grid_xy.shape)
        assert len(rgb.shape) == input_ndims

        if input_ndims > 1 and input_ndims < 5:
            for i in range(5 - input_ndims):
                grid_xy = grid_xy.unsqueeze(1)
                rgb = rgb.unsqueeze(1)
            assert idx is not None
        elif input_ndims != 5:
            raise ValueError("Bilateral grid slicing only takes either 2D, 3D, 4D and 5D inputs")

        grids = self.grids
        if idx is not None:
            grids = grids[idx]
        assert grids.shape[0] == grid_xy.shape[0]

        grid_xy = (grid_xy - 0.5) * 2
        grid_z = self.rgb2gray(rgb)
        grid_xyz = torch.cat([grid_xy, grid_z], dim=-1)

        affine_mats = F.grid_sample(grids, grid_xyz, mode="bilinear", align_corners=True, padding_mode="border")
        affine_mats = affine_mats.permute(0, 2, 3, 4, 1)
        affine_mats = affine_mats.reshape(*affine_mats.shape[:-1], 3, 4)

        for _ in range(5 - input_ndims):
            affine_mats = affine_mats.squeeze(1)

        return affine_mats


try:
    from fused_bilagrid import BilateralGrid as _Grid, slice as _slice
    FUSED = True
except ImportError:
    _Grid, _slice = BilateralGrid, slice
    FUSED = False


class BilateralGridModule:
    """Per-view bilateral grids with their own Adam and the gsplat lr schedule
    (linear warmup from 0.01x over 1000 its, exponential decay to 0.01x at total_iterations).

    Regulariser (reg_loss) = tv_weight * TV + identity_weight * mean((G_v - I)^2) + mean_weight * mean((G_v - mean_v G)^2).
    The last two keep the raw render (eval, export, finetune) from drifting per view (HANDOFF 7.1); both default to 0.
    until >= 0: the grid is used up to and including that iteration only (is_active), later its are grid-free.
    """

    def __init__(self, num_views, grid_X=16, grid_Y=16, grid_W=8, lr=2e-3, tv_weight=10.0,
                 total_iterations=30000, device="cuda", identity_weight=0.0, mean_weight=0.0, until=-1):
        self.grid = _Grid(num_views, grid_X=grid_X, grid_Y=grid_Y, grid_W=grid_W).to(device)
        self.lr = lr
        self.tv_weight = tv_weight
        self.identity_weight = identity_weight
        self.mean_weight = mean_weight
        self.until = until
        self.total_iterations = total_iterations
        self.optimizer = torch.optim.Adam(self.grid.parameters(), lr=lr, eps=1e-15)
        self._xy = None
        # (1, 12, 1, 1, 1) identity affine (row-major 3x4), broadcast over the grids (N, 12, L, H, W)
        self._identity = torch.tensor([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]).to(self.grid.grids).view(1, 12, 1, 1, 1)

    def _grid_xy(self, height, width, device):
        if self._xy is None or self._xy.shape[1:3] != (height, width) or self._xy.device != device:
            y, x = torch.meshgrid(torch.arange(height, device=device) + 0.5,
                                  torch.arange(width, device=device) + 0.5, indexing="ij")
            self._xy = torch.stack([x / width, y / height], dim=-1)[None]
        return self._xy

    def apply(self, image, view_idx):
        """image (3, H, W) -> view_idx's grid applied, (3, H, W)."""
        _, height, width = image.shape
        rgb = image.permute(1, 2, 0)[None]
        idx = torch.full((1, 1), int(view_idx), dtype=torch.long, device=image.device)
        out = _slice(self.grid, self._grid_xy(height, width, image.device), rgb, idx)["rgb"]
        return out[0].permute(2, 0, 1)

    def tv_loss(self):
        return self.tv_weight * total_variation_loss(self.grid.grids)

    def identity_loss(self):
        """Unweighted mean squared deviation of every grid entry from the identity affine (0 at init)."""
        return (self.grid.grids - self._identity).pow(2).mean()

    def mean_loss(self):
        """Unweighted mean squared deviation of each view's grid from the mean grid over views."""
        grids = self.grid.grids
        return (grids - grids.mean(0, keepdim=True)).pow(2).mean()

    def reg_loss(self):
        """tv_weight * TV + identity_weight * identity_loss + mean_weight * mean_loss (zero terms skipped)."""
        loss = self.tv_loss()
        if self.identity_weight > 0:
            loss = loss + self.identity_weight * self.identity_loss()
        if self.mean_weight > 0:
            loss = loss + self.mean_weight * self.mean_loss()
        return loss

    def is_active(self, iteration):
        """Grid applied (and stepped) at this iteration: always when until < 0, else iteration <= until."""
        return self.until < 0 or iteration <= self.until

    @torch.no_grad()
    def drift_stats(self):
        """Diagnostics for gs_stats.json: RMS deviation from identity and RMS spread across views."""
        return {"identity_rms": float(self.identity_loss().sqrt()), "view_spread_rms": float(self.mean_loss().sqrt())}

    def lr_at(self, iteration):
        warmup = 0.01 + 0.99 * min(iteration, 1000) / 1000
        return self.lr * warmup * 0.01 ** (iteration / self.total_iterations)

    def update_lr(self, iteration):
        lr = self.lr_at(iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"grid": self.grid.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.grid.load_state_dict(state["grid"])
        self.optimizer.load_state_dict(state["optimizer"])

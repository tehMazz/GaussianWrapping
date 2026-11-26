"""Multi-view mask consensus (BEA-GS `reprojection_batch`).

For every view, the masks of all views are warped into it through the rendered depth.
A warp is kept only where the forward/backward reprojection agrees; the new mask is the
average of the kept warps, per channel, over views where that channel is present.
Where no warp is kept (no view, not even the target itself, passes the check, e.g. rendered depth 0),
fallback="own" keeps the target's own input mask; "zero" writes 0 there (the old behaviour, which
labelled about 32% of the spinnerf_7 object as background and drove the finetune's SAG erosion).
"""
import torch
import torch.nn.functional as F

from utils.graphics_utils import fov2focal


def _intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    return fov2focal(cam.FoVx, W), fov2focal(cam.FoVy, H), (W - 1) / 2, (H - 1) / 2


def _pixel_coords(H, W):
    ys, xs = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
    return torch.stack([xs, ys], dim=-1).float()  # (H, W, 2)


def backproject(depth, cam):
    """(H, W) depth -> (H, W, 3) world points."""
    fx, fy, cx, cy = _intrinsics(cam)
    uv = _pixel_coords(*depth.shape)
    cam_pts = torch.stack([(uv[..., 0] - cx) / fx * depth, (uv[..., 1] - cy) / fy * depth, depth, torch.ones_like(depth)], -1)
    return (cam_pts @ cam.world_view_transform.inverse())[..., :3]


def project(points, cam):
    """(..., 3) world points -> (..., 2) pixel coords and (...) camera depth."""
    fx, fy, cx, cy = _intrinsics(cam)
    p = torch.cat([points, torch.ones_like(points[..., :1])], -1) @ cam.world_view_transform
    z = p[..., 2]
    zs = z.clamp_min(1e-6)
    return torch.stack([p[..., 0] / zs * fx + cx, p[..., 1] / zs * fy + cy], -1), z


def _to_grid(uv, H, W):
    """Pixel coords -> grid_sample coords (align_corners=False)."""
    return torch.stack([(2 * uv[..., 0] + 1) / W - 1, (2 * uv[..., 1] + 1) / H - 1], -1)


@torch.no_grad()
def iter_reproject_masks(cameras, depths, masks, present, threshold=0.01, chunk=4, fallback="own"):
    """Consensus masks.

    Depths and masks may remain on CPU; only one target view and a small reference
    chunk are moved to CUDA at a time. Yields one (C, H, W) float32 CPU result.
    """
    N = len(cameras)
    C = present.shape[1]
    H, W = depths[0].shape
    pix = _to_grid(_pixel_coords(H, W), H, W)                                      # (H, W, 2)
    limit = 1 - 1 / torch.tensor([W, H], device="cuda")
    inside = lambda g: (g.abs() < limit).all(-1)
    present = present.cuda()
    for i, cur in enumerate(cameras):
        cur_depth = depths[i].to(device="cuda", dtype=torch.float32)
        cur_world = backproject(cur_depth, cur)
        votes = torch.zeros((C, H, W), device="cuda")
        count = torch.zeros((C, H, W), device="cuda")
        for s in range(0, N, chunk):
            refs = cameras[s:s + chunk]
            ref_depths = torch.stack([d.to(device="cuda", dtype=torch.float32)
                                      for d in depths[s:s + len(refs)]])
            fwd, back, ok, warped_masks = [], [], [], []
            for r, ref in enumerate(refs):
                ref_world = backproject(ref_depths[r], ref)
                uv, z = project(cur_world, ref)                                    # cur pixels seen from ref
                uv_back, _ = project(ref_world, cur)                               # ref pixels seen from cur
                fwd.append(_to_grid(uv, H, W))
                back.append(_to_grid(uv_back, H, W))
                ok.append((z > 1e-4) & (cur_depth > 0))
                mask = masks[s + r]
                if mask is None:
                    mask = torch.zeros((C, H, W), dtype=torch.float32)
                warped_masks.append(mask.to(device="cuda", dtype=torch.float32))
            fwd, back, ok = torch.stack(fwd), torch.stack(back), torch.stack(ok)
            back = back.masked_fill((~inside(back) | (ref_depths <= 0))[..., None], 10.0)
            round_trip = F.grid_sample(back.permute(0, 3, 1, 2), fwd, mode="nearest", align_corners=False).permute(0, 2, 3, 1)
            valid = ok & inside(fwd) & ((round_trip - pix).norm(dim=-1) < threshold)  # (B, H, W)
            warped = F.grid_sample(torch.stack(warped_masks), fwd, mode="nearest", align_corners=False)
            votes += (warped * valid[:, None]).sum(0)
            count += (valid[:, None] & present[s:s + len(refs), :, None, None]).sum(0)
        own = masks[i]
        if fallback == "own" and own is not None:
            empty = own.to(device="cuda", dtype=torch.float32)
        elif fallback in ("own", "zero"):
            empty = torch.zeros_like(votes)
        else:
            raise ValueError(f"fallback must be 'own' or 'zero', got {fallback!r}")
        yield torch.where(count > 0, votes / count.clamp_min(1), empty).cpu()


@torch.no_grad()
def reproject_masks(cameras, depths, masks, present, threshold=0.01, chunk=4, fallback="own"):
    """Compatibility wrapper returning all consensus masks as a CPU tensor."""
    if isinstance(masks, torch.Tensor):
        mask_views = list(masks.unbind(0))
    else:
        mask_views = masks
    return torch.stack(list(iter_reproject_masks(cameras, depths, mask_views, present, threshold, chunk, fallback)))

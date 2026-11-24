"""Render a standard 3DGS PLY with Gaussian Wrapping's own rasterizer (render_ours) into ArtiFixer-GS's render layout.

    python gaussian_wrapping/render_ply_frames.py --ply <ply> --trajectory <transforms.json> --out_dir <dir> [--depth]
        [--kernel_size 0] [--white_background]

ArtiFixer-GS (submodules/artifixer_gs, artifixer_gs.render.render_ply) runs this as RENDER_CMD instead of gsplat, so
every ArtiFixer render (prepare, removal background / deleted, ArtiFixer3D+) is GW-native; ArtiFixer itself only does the
diffusion. Same contract as artifixer_gs.render.render_frames:
    <out_dir>/renders/{i:05d}.png   RGB composited on black (white with --white_background)
    <out_dir>/opacity/{i:05d}.png   accumulated alpha * 255
    <out_dir>/depth/{i:05d}.npy     --depth: z-depth (float32) where alpha >= 0.5, else 0: GW's median depth, or the
                                    expected depth (sum w z / alpha, as gsplat's RGB+ED) where GW has no median (alpha
                                    in [0.5, 0.55), or no T = 0.5 crossing within SAMPLE_RANGE). artifixer_gs.hole_mesh
                                    compares the mesh with it wherever the background is opaque
The PLY is export_3dgs_ply.py's (3D filter baked into scale and opacity), or any subset of it (removal.py's removed.ply /
deleted.ply): it has no filter_3D, so GaussianModel.load_ply uses its scale and opacity as they are, which is exactly
GW's filtered model. --kernel_size: the 2D kernel (0: train.py renders with 0.0).
Cameras come from the transforms.json (artifixer_gs.render.trajectory_cameras: OpenCV w2c + K) as GW cameras with
the principal point at the image centre, as GW trains them (scene/dataset_readers.py); an off-centre principal point
or lens distortion is reported once and ignored (artifixer_gs.render.check_pinhole: IGNORE_DISTORTION=1).
Needs the ArtiFixer-GS repo on PYTHONPATH (render_ply passes it).
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arguments import PipelineParams
from gaussian_renderer.ours import render_ours
from scene import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov


def gw_camera(viewmat, K, width, height, uid):
    """GW Camera of an OpenCV world-to-camera (4, 4) and intrinsics (3, 3); principal point at the centre."""
    w2c = np.asarray(viewmat, dtype=np.float64)
    fx, fy = float(K[0][0]), float(K[1][1])
    image = torch.zeros(3, int(height), int(width))
    return Camera(colmap_id=uid, R=w2c[:3, :3].T, T=w2c[:3, 3], FoVx=focal2fov(fx, width), FoVy=focal2fov(fy, height),
                  image=image, gt_alpha_mask=None, image_name=f"{uid:05d}", uid=uid, data_device="cpu")


def principal_point_offset(Ks, width, height):
    """Largest |c - size / 2| in pixels over the frames (GW renders with c = size / 2)."""
    Ks = np.asarray(Ks, dtype=np.float64)
    return float(max(np.abs(Ks[:, 0, 2] - width / 2).max(), np.abs(Ks[:, 1, 2] - height / 2).max()))


def ply_vertex_count(path):
    with open(path, "rb") as f:
        for raw in iter(f.readline, b""):
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"{path}: no 'element vertex' in the PLY header")


def load_ply(path):
    gaussians = GaussianModel(None)  # SH degree from the PLY's f_rest count
    gaussians.load_ply(str(path))
    assert not gaussians.use_mip_filter, f"{path} has filter_3D: render the export_3dgs_ply.py PLY, not the GW one"
    return gaussians


def to_png_rgb(rgb):
    """(3, H, W) float -> (H, W, 3) uint8, quantized as artifixer_gs.render (rgb * 255 + 0.5)."""
    return (rgb.permute(1, 2, 0) * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()


def to_png_alpha(alpha):
    return (alpha * 255).round().clamp(0, 255).byte().cpu().numpy()


def view_z(cam, xyz):
    """[P] camera-space z of world points (GW row-vector world_view_transform)."""
    return xyz @ cam.world_view_transform[:3, 2] + cam.world_view_transform[3, 2]


def expected_depth(cam, gaussians, pipe, kernel_size, alpha):
    """[H, W] sum(w z) / alpha: a render with the Gaussians' view z as colour, on black."""
    z = view_z(cam, gaussians.get_xyz)[:, None].repeat(1, 3)
    out = render_ours(cam, gaussians, pipe, torch.zeros(3, device="cuda"), kernel_size=kernel_size, require_depth=False,
                      colors_precomp=z)
    return out["render"][0] / alpha.clamp_min(1e-6)


def surface_depth(median, alpha, expected):
    """Median depth where GW has one, else the expected depth; 0 where alpha < 0.5 (not a surface)."""
    z = torch.where(median > 0, median, expected)
    return torch.where(alpha < 0.5, torch.zeros_like(z), z)


@torch.no_grad()
def render_frames(ply, trajectory, out_dir, pipe, depth=False, kernel_size=0.0, white_background=False):
    from artifixer_gs.render import trajectory_cameras  # the ArtiFixer-GS repo is on PYTHONPATH

    viewmats, Ks, width, height = trajectory_cameras(Path(trajectory))
    offset = principal_point_offset(Ks, width, height)
    if offset > 0.5:
        print(f"[render_ply_frames] WARNING: principal point up to {offset:.2f} px off the image centre in {trajectory}; "
              f"GW renders (and trained) with it centred: set CENTER_PRINCIPAL_POINT=1", file=sys.stderr)
    out_dir = Path(out_dir)
    for name in ("renders", "opacity", *(("depth",) if depth else ())):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    if ply_vertex_count(ply) == 0:  # e.g. an empty removal subset: nothing to rasterize
        print(f"[render_ply_frames] {ply} has no Gaussians: background frames", flush=True)
        rgb = np.full((height, width, 3), 255 if white_background else 0, np.uint8)
        for index in range(len(viewmats)):
            Image.fromarray(rgb).save(out_dir / "renders" / f"{index:05d}.png")
            Image.fromarray(np.zeros((height, width), np.uint8)).save(out_dir / "opacity" / f"{index:05d}.png")
            if depth:
                np.save(out_dir / "depth" / f"{index:05d}.npy", np.zeros((height, width), np.float32))
        return len(viewmats)
    gaussians = load_ply(ply)
    bg = torch.tensor([1., 1., 1.] if white_background else [0., 0., 0.], device="cuda")
    print(f"[render_ply_frames] {len(viewmats)} frames of {trajectory} from {ply} ({gaussians.get_xyz.shape[0]} Gaussians, "
          f"SH {gaussians.max_sh_degree}, {width}x{height}, kernel_size {kernel_size}): render_ours", flush=True)
    for index in range(len(viewmats)):
        cam = gw_camera(viewmats[index], Ks[index], width, height, index)
        out = render_ours(cam, gaussians, pipe, bg, kernel_size=kernel_size, require_depth=depth)
        alpha = out["alpha"][0]
        Image.fromarray(to_png_rgb(out["render"])).save(out_dir / "renders" / f"{index:05d}.png")
        Image.fromarray(to_png_alpha(alpha)).save(out_dir / "opacity" / f"{index:05d}.png")
        if depth:
            z = surface_depth(out["median_depth"][0], alpha, expected_depth(cam, gaussians, pipe, kernel_size, alpha))
            np.save(out_dir / "depth" / f"{index:05d}.npy", z.float().cpu().numpy())
    return len(viewmats)


if __name__ == "__main__":
    parser = ArgumentParser(description="ArtiFixer-GS render_ply backend: a 3DGS PLY rendered with GW's render_ours")
    pipeline = PipelineParams(parser)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--trajectory", required=True, help="transforms.json-style camera path (OpenGL c2w)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--depth", action="store_true", help="also write depth/{i:05d}.npy (median z-depth)")
    parser.add_argument("--kernel_size", type=float, default=0.0, help="2D kernel of the trained model (cfg_args)")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--eps2d", type=float, default=None, help="ignored (gsplat's flag; GW uses --kernel_size)")
    args = parser.parse_args()
    render_frames(args.ply, args.trajectory, args.out_dir, pipeline.extract(args), args.depth, args.kernel_size,
                  args.white_background)

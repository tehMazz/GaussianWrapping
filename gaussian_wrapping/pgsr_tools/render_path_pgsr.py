"""Render a PGSR model along a camera path from make_camera_path.py (same outputs as render_path_gw.py).

    cd /leonardo_work/IscrC_EditGS/repos/PGSR && python <this file> --ply <model>/point_cloud/iteration_30000/point_cloud.ply \
        --path <path.json> --out <dir> [--max_frames K]

PYTHONPATH must hold gaussian_wrapping/pgsr_tools/shim and the built diff-plane-rasterization (slurm/pgsr_scene.sh);
the cwd must be the PGSR repo root so its packages (scene, gaussian_renderer, ...) import, not GW's.
Writes <out>/rgb/%05d.png, <out>/depth/%05d.npy (float16 plane z-depth), <out>/normal/%05d.png (PGSR's rendered
normal, camera space, facing the camera, (n+1)/2). Black background, no appearance model, as in PGSR's render.py.
"""
import json
import os
import sys
from argparse import ArgumentParser, Namespace

sys.path.insert(0, os.getcwd())          # the PGSR repo root

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


def main():
    ap = ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--sh_degree", type=int, default=3)
    a = ap.parse_args()
    P = json.load(open(a.path))
    W, H = P["width"], P["height"]
    gaussians = GaussianModel(a.sh_degree)
    gaussians.load_ply(a.ply)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    for sub in ("rgb", "depth", "normal"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    frames = P["frames"][:a.max_frames] if a.max_frames else P["frames"]
    with torch.no_grad():
        for i, f in enumerate(frames):
            w2c = np.array(f["w2c"])
            cam = Camera(colmap_id=i, R=w2c[:3, :3].T, T=w2c[:3, 3], FoVx=P["fovx"], FoVy=P["fovy"],
                         image_width=W, image_height=H, image_path=None, image_name=f"{i:05d}", uid=i,
                         preload_img=False, data_device="cpu")
            out = render(cam, gaussians, pipe, bg, app_model=None, return_plane=True, return_depth_normal=False)
            torchvision.utils.save_image(out["render"].clamp(0, 1), os.path.join(a.out, "rgb", f"{i:05d}.png"))
            depth = torch.nan_to_num(out["plane_depth"][0], 0.0).clamp_min(0)
            np.save(os.path.join(a.out, "depth", f"{i:05d}.npy"), depth.cpu().numpy().astype(np.float16))
            n = F.normalize(out["rendered_normal"], dim=0)
            n = torch.where(n[2:3] > 0, -n, n)
            torchvision.utils.save_image(n * 0.5 + 0.5, os.path.join(a.out, "normal", f"{i:05d}.png"))
    json.dump({"frames": len(frames), "ply": a.ply}, open(os.path.join(a.out, "render_info.json"), "w"))
    print(f"[path] {len(frames)} frames -> {a.out}")


if __name__ == "__main__":
    main()

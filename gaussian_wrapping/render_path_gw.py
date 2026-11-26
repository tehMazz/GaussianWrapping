"""Render a GW model (and optionally a mesh) along a camera path from make_camera_path.py.

    python gaussian_wrapping/render_path_gw.py -m <GW model> --iteration N --path <path.json> --out <dir>
           [--mesh <mesh.ply>] [--max_frames K]

Writes <out>/rgb/%05d.png, <out>/depth/%05d.npy (float16 z-depth, 0 = none), <out>/normal/%05d.png (camera space,
facing the camera, (n+1)/2) and, with --mesh, <out>/mesh/%05d.png (grey, shaded by facing; nvdiffrast).
No appearance model / bilateral grid: novel views have no per-image correction. Every camera gets zfar = 1e4 so far
scenes (SPIn-NeRF at 115-130 units) are not clipped in the mesh render. The median-depth rasterizer build follows
PYTHONPATH (RASTER_PATH in slurm/common.sh).
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from beags.common import load_model, parse_args
from gaussian_renderer.ours import render_ours
from scene.cameras import Camera


def add_args(parser):
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mesh", default="")
    parser.add_argument("--max_frames", type=int, default=0)


def path_cameras(path, max_frames=0):
    P = json.load(open(path))
    W, H = P["width"], P["height"]
    frames = P["frames"][:max_frames] if max_frames else P["frames"]
    for i, f in enumerate(frames):
        w2c = np.array(f["w2c"])
        yield i, Camera(colmap_id=i, R=w2c[:3, :3].T, T=w2c[:3, 3], FoVx=P["fovx"], FoVy=P["fovy"],
                        image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=f"{i:05d}", uid=i,
                        data_device="cpu", zfar=1e4)


def facing_normal(n_cam):
    """(3, H, W) camera-space normals, flipped to face the camera (camera looks along +z)."""
    n = F.normalize(n_cam, dim=0)
    return torch.where(n[2:3] > 0, -n, n)


@torch.no_grad()
def main():
    dataset, pipe, opt, args = parse_args("render GW along a path", add_args)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=False)
    for sub in ("rgb", "depth", "normal") + (("mesh",) if args.mesh else ()):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    mesh_ctx = None
    if args.mesh:
        import nvdiffrast.torch as dr
        import trimesh
        from render_mesh_views import rasterize
        m = trimesh.load(args.mesh, process=False)
        verts = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
        faces = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device="cuda")
        face_n = F.normalize(torch.cross(verts[faces[:, 1].long()] - verts[faces[:, 0].long()],
                                         verts[faces[:, 2].long()] - verts[faces[:, 0].long()], dim=-1), dim=-1)
        mesh_ctx = (dr.RasterizeCudaContext(), verts, faces, face_n, rasterize)
        print(f"[path] mesh {args.mesh}: {verts.shape[0]} vertices, {faces.shape[0]} faces")
    bg = torch.zeros(3, device="cuda")
    n = 0
    for i, cam in path_cameras(args.path, args.max_frames):
        pkg = render_ours(cam, gaussians, pipe, bg, require_depth=True)
        torchvision.utils.save_image(pkg["render"].clamp(0, 1), os.path.join(args.out, "rgb", f"{i:05d}.png"))
        depth = torch.nan_to_num(pkg["median_depth"][0], 0.0).clamp_min(0)
        np.save(os.path.join(args.out, "depth", f"{i:05d}.npy"), depth.cpu().numpy().astype(np.float16))
        torchvision.utils.save_image(facing_normal(pkg["normal"]) * 0.5 + 0.5,
                                     os.path.join(args.out, "normal", f"{i:05d}.png"))
        if mesh_ctx is not None:
            glctx, verts, faces, face_n, rasterize = mesh_ctx
            img, _ = rasterize(glctx, cam, verts, faces, face_n)
            torchvision.utils.save_image(img, os.path.join(args.out, "mesh", f"{i:05d}.png"))
        n += 1
    json.dump({"frames": n, "model": dataset.model_path, "iteration": scene.loaded_iter, "mesh": args.mesh},
              open(os.path.join(args.out, "render_info.json"), "w"))
    print(f"[path] {n} frames -> {args.out}")


if __name__ == "__main__":
    main()

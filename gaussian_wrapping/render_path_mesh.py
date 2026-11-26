"""Render any mesh along a camera path from make_camera_path.py (grey, shaded by facing; nvdiffrast, zfar 1e4).

    python gaussian_wrapping/render_path_mesh.py --mesh <mesh.ply> --path <path.json> --out <dir> [--max_frames K]

Writes <out>/mesh/%05d.png (white where no face). Needs no model, so it serves PGSR's TSDF mesh as well as GW's.
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), 'submodules'))

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import torchvision
import trimesh

from render_mesh_views import rasterize
from render_path_gw import path_cameras


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    a = ap.parse_args()
    m = trimesh.load(a.mesh, process=False)
    verts = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
    faces = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device="cuda")
    face_n = F.normalize(torch.cross(verts[faces[:, 1].long()] - verts[faces[:, 0].long()],
                                     verts[faces[:, 2].long()] - verts[faces[:, 0].long()], dim=-1), dim=-1)
    print(f"[mesh] {a.mesh}: {verts.shape[0]} vertices, {faces.shape[0]} faces")
    os.makedirs(os.path.join(a.out, "mesh"), exist_ok=True)
    glctx = dr.RasterizeCudaContext()
    n = 0
    for i, cam in path_cameras(a.path, a.max_frames):
        img, _ = rasterize(glctx, cam, verts, faces, face_n)
        torchvision.utils.save_image(img, os.path.join(a.out, "mesh", f"{i:05d}.png"))
        n += 1
    print(f"[mesh] {n} frames -> {a.out}/mesh")


if __name__ == "__main__":
    main()

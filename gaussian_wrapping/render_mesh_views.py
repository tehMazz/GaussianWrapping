"""Render a mesh from the training/test cameras of a GW model, to look at it next to the photos.

    python gaussian_wrapping/render_mesh_views.py -m <model> --iteration N --mesh <mesh.ply> [--n_views 8] [--out <dir>]

Views: --n_views cameras spread evenly over test + train (the beags_render_labels.py rule, so the views match).
Per view, <out>/grid/<view>.png = photo | mesh (grey, shaded by facing) | overlay | mesh with the camera's stock far plane.
The stock Camera has zfar = 100 (scene/cameras.py), and nvdiffrast clips everything farther than that, which on
forward-facing scenes (SPIn-NeRF: object ~116 units away) removes the object. Here every camera is re-projected with
zfar = 2 x the farthest mesh vertex it sees (--zfar to force a value). Prints mesh coverage per view for both.
"""
import copy
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import torchvision
import trimesh

from beags.common import load_model, parse_args
from utils.graphics_utils import getProjectionMatrix
from utils.grid_titles import add_column_titles


def add_args(parser):
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--n_views", default=8, type=int)
    parser.add_argument("--out", default=None)
    parser.add_argument("--zfar", default=0.0, type=float, help="far plane for the fixed render (0 = auto per camera)")


def with_far(cam, zfar):
    c = copy.copy(cam)
    c.zfar = zfar
    c.projection_matrix = getProjectionMatrix(znear=c.znear, zfar=zfar, fovX=c.FoVx, fovY=c.FoVy).transpose(0, 1).cuda()
    c.full_proj_transform = c.world_view_transform.unsqueeze(0).bmm(c.projection_matrix.unsqueeze(0)).squeeze(0)
    return c


def rasterize(glctx, cam, verts, faces, face_n):
    pos = torch.cat([verts, torch.ones_like(verts[:, :1])], 1) @ cam.full_proj_transform
    rast, _ = dr.rasterize(glctx, pos[None], faces, resolution=[cam.image_height, cam.image_width])
    tri = rast[0, ..., 3].long() - 1                                   # -1 = no face
    hit = tri >= 0
    cam_center = cam.camera_center
    n = face_n[tri.clamp_min(0)]                                       # (H, W, 3)
    p = dr.interpolate(verts[None], rast, faces)[0][0]                 # world position per pixel
    view = F.normalize(cam_center - p, dim=-1)
    shade = (n * view).sum(-1).abs().clamp(0, 1)
    img = torch.where(hit, 0.15 + 0.8 * shade, torch.ones_like(shade))
    return img[None].expand(3, -1, -1), hit


@torch.no_grad()
def main():
    dataset, pipe, opt, args = parse_args("render mesh views", add_args)
    scene, _ = load_model(dataset, args.iteration, with_labels=False)
    mesh = trimesh.load(args.mesh, process=False)
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device="cuda")
    face_n = F.normalize(torch.cross(verts[faces[:, 1].long()] - verts[faces[:, 0].long()],
                                     verts[faces[:, 2].long()] - verts[faces[:, 0].long()], dim=-1), dim=-1)
    print(f"[mesh] {args.mesh}: {verts.shape[0]} vertices, {faces.shape[0]} faces")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.mesh)), "mesh_views")
    os.makedirs(os.path.join(out, "grid"), exist_ok=True)
    cams = list(scene.getTestCameras()) + list(scene.getTrainCameras())
    cams = [cams[i] for i in np.unique(np.linspace(0, len(cams) - 1, min(args.n_views, len(cams))).round().astype(int))]
    glctx = dr.RasterizeCudaContext()
    rows, small = [], []
    titles = ["photo", "mesh (far plane fixed)", "mesh over photo", f"mesh (camera far = {cams[0].zfar:g})"]
    for cam in cams:
        z = (torch.cat([verts, torch.ones_like(verts[:, :1])], 1) @ cam.world_view_transform)[:, 2]
        zfar = args.zfar or float(2 * z.max().clamp_min(1.0))
        fixed, hit_fixed = rasterize(glctx, with_far(cam, zfar), verts, faces, face_n)
        stock, hit_stock = rasterize(glctx, cam, verts, faces, face_n)
        photo = cam.original_image[:3].cuda()
        overlay = torch.where(hit_fixed[None], 0.45 * photo + 0.55 * fixed, photo)
        grid = torch.cat([photo, fixed, overlay, stock], dim=2)
        torchvision.utils.save_image(grid, os.path.join(out, "grid", f"{cam.image_name}.png"))
        add_column_titles(os.path.join(out, "grid", f"{cam.image_name}.png"), titles)
        small.append(F.interpolate(grid[None], scale_factor=0.5, mode="area")[0].cpu())
        row = {"view": cam.image_name, "zfar_fixed": zfar, "zfar_stock": cam.zfar,
               "coverage_fixed": float(hit_fixed.float().mean()), "coverage_stock": float(hit_stock.float().mean()),
               "mesh_depth_p50": float(z[z > 0].median()) if bool((z > 0).any()) else None}
        rows.append(row)
        print(f"[mesh] {json.dumps(row)}")
    width = min(g.shape[2] for g in small)
    torchvision.utils.save_image(torch.cat([g[:, :, :width] for g in small], dim=1), os.path.join(out, "grid_all.jpg"))
    add_column_titles(os.path.join(out, "grid_all.jpg"), titles)
    with open(os.path.join(out, "coverage.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[mesh] {len(cams)} views -> {out}/grid (photo | mesh, far fixed | overlay | mesh, stock far 100)")


if __name__ == "__main__":
    main()

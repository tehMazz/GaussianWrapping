"""
Export a Gaussian Wrapping model as a standard INRIA 3DGS PLY with the 3D (mip) filter baked in.

    python gaussian_wrapping/export_3dgs_ply.py -m <model> --iteration N [--out <ply>] [--check] [--split_labels]

The baked parameters are exactly GaussianModel.get_scaling_with_3D_filter / get_opacity_with_3D_filter:
    scale'   = log sqrt(s^2 + f^2)
    opacity' = logit(sigmoid(o) * sqrt(prod(s^2) / prod(s^2 + f^2)))
Normals are zero; filter_3D, occupancies, Gaussian features and confidence are dropped.
The default output is <model>/../export/point_cloud.ply (flat layout, next to the model dir, not inside it),
with export_info.json in the same directory.
--split_labels also writes label_<k>.ply (the Gaussians with BEA-GS label k, from labels.pt next to the input PLY)
next to the output, for viewing each label on its own.
--check (GPU) renders the --check_splits views (train,test; every --check_stride-th) with render_ours at the exported
iteration and with gsplat from the exported PLY, and writes <export dir>/gsplat_check.json (gsplat_check.py: per-view
and mean PSNR gsplat vs native, alpha MAE, PSNR of each vs the photos; pass = drop <= --max_drop, 0.5 dB).
gsplat importable here: in-process. Otherwise (the GW env) the native views go to --check_dir (default
<export dir>/gsplat_check_views, removed unless --keep_check_views) and gsplat_check.py runs in --gs_python
(default $GS_RENDER_PY, e.g. /leonardo_scratch/fast/IscrC_EditGS/envs/gsplat/bin/python); without either it skips.
--check_dir must be new, empty or a previous dump (only manifest.json + *.npz), else the check refuses to clear it.
A failing check never fails the export: export_info.json["check"] = {"error": ...} and no gsplat_check.json.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)

import json
import math
import subprocess
from argparse import ArgumentParser
from typing import Dict, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

import gsplat_check


def _stack_properties(element, prefix: str) -> np.ndarray:
    names = [p.name for p in element.properties if p.name.startswith(prefix)]
    names = sorted(names, key=lambda x: int(x.split('_')[-1]))
    if len(names) == 0:
        return np.zeros((element.count, 0), dtype=np.float32)
    return np.stack([np.asarray(element[name]) for name in names], axis=1).astype(np.float32)


def read_gw_ply(path: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Read the raw (non-activated) parameters of a Gaussian Wrapping PLY.

    Returns:
        Dict with xyz (N, 3), f_dc (N, 3), f_rest (N, 3 * (K - 1)) channel-major as in INRIA 3DGS,
        opacity (N, 1) logits, scaling (N, 3) log-scales, rotation (N, 4) unnormalized wxyz quaternions,
        and filter_3D (N, 1) or None.
    """
    element = PlyData.read(path).elements[0]
    property_names = [p.name for p in element.properties]
    return {
        "xyz": np.stack([np.asarray(element[k]) for k in ("x", "y", "z")], axis=1).astype(np.float32),
        "f_dc": _stack_properties(element, "f_dc_"),
        "f_rest": _stack_properties(element, "f_rest_"),
        "opacity": np.asarray(element["opacity"]).astype(np.float32)[..., None],
        "scaling": _stack_properties(element, "scale_"),
        "rotation": _stack_properties(element, "rot_"),
        "filter_3D": np.asarray(element["filter_3D"]).astype(np.float32)[..., None] if "filter_3D" in property_names else None,
    }


def bake_3d_filter(scaling: torch.Tensor, opacity: torch.Tensor, filter_3D: torch.Tensor):
    """
    Bake the 3D filter into the raw parameters, following GaussianModel.get_scaling_n_opacity_with_3D_filter.
    The math runs in float64 so that the logit stays finite for opacities close to 1.

    Args:
        scaling (torch.Tensor): Log-scales. Has shape (N, 3).
        opacity (torch.Tensor): Opacity logits. Has shape (N, 1).
        filter_3D (torch.Tensor): 3D filter std. Has shape (N, 1).

    Returns:
        scaling (torch.Tensor): Baked log-scales. Has shape (N, 3).
        opacity (torch.Tensor): Baked opacity logits. Has shape (N, 1).
    """
    scales_square = torch.square(torch.exp(scaling.double()))
    scales_after_square = scales_square + torch.square(filter_3D.double())
    det1 = scales_square.prod(dim=1)
    det2 = scales_after_square.prod(dim=1)
    coef = torch.sqrt(det1 / det2)
    baked_opacity = (torch.sigmoid(opacity.double()) * coef[..., None]).clamp(min=1e-12, max=1. - 1e-12)
    baked_scaling = torch.log(torch.sqrt(scales_after_square))
    return baked_scaling.to(scaling.dtype), torch.logit(baked_opacity).to(opacity.dtype)


def write_3dgs_ply(path: str, gaussians: Dict[str, np.ndarray]):
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attributes += ['f_dc_{}'.format(i) for i in range(gaussians["f_dc"].shape[1])]
    attributes += ['f_rest_{}'.format(i) for i in range(gaussians["f_rest"].shape[1])]
    attributes += ['opacity']
    attributes += ['scale_{}'.format(i) for i in range(gaussians["scaling"].shape[1])]
    attributes += ['rot_{}'.format(i) for i in range(gaussians["rotation"].shape[1])]
    values = np.concatenate(
        [
            gaussians["xyz"], np.zeros_like(gaussians["xyz"]), gaussians["f_dc"], gaussians["f_rest"],
            gaussians["opacity"], gaussians["scaling"], gaussians["rotation"],
        ],
        axis=1,
    ).astype(np.float32)
    assert values.shape[1] == len(attributes)
    elements = np.empty(values.shape[0], dtype=[(attribute, 'f4') for attribute in attributes])
    for i, attribute in enumerate(attributes):
        elements[attribute] = values[:, i]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def export_3dgs_ply(ply_in: str, ply_out: str, split_labels: bool = False) -> Dict:
    """
    Bake the 3D filter of a Gaussian Wrapping PLY and write a standard 3DGS PLY. Returns the export info.
    With split_labels, also write label_<k>.ply per label of labels.pt (next to ply_in) in the directory of ply_out.
    """
    gaussians = read_gw_ply(ply_in)
    has_filter_3D = gaussians["filter_3D"] is not None
    if has_filter_3D:
        scaling, opacity = bake_3d_filter(
            torch.from_numpy(gaussians["scaling"]),
            torch.from_numpy(gaussians["opacity"]),
            torch.from_numpy(gaussians["filter_3D"]),
        )
        gaussians["scaling"], gaussians["opacity"] = scaling.numpy(), opacity.numpy()
    write_3dgs_ply(ply_out, gaussians)

    label_counts = None
    if split_labels:
        labels_path = os.path.join(os.path.dirname(ply_in), "labels.pt")
        assert os.path.exists(labels_path), f"--split_labels: {labels_path} not found"
        labels = torch.load(labels_path, map_location="cpu").numpy()
        assert labels.shape[0] == gaussians["xyz"].shape[0], "labels.pt does not match the PLY"
        label_counts = {}
        for k in range(int(labels.max()) + 1):
            keep = labels == k
            path = os.path.join(os.path.dirname(os.path.abspath(ply_out)), f"label_{k}.ply")
            write_3dgs_ply(path, {name: value[keep] for name, value in gaussians.items() if value is not None})
            label_counts[k] = int(keep.sum())
            print(f"[INFO] Saved {label_counts[k]} Gaussians of label {k} to {path}")

    n_sh_coeffs = 1 + gaussians["f_rest"].shape[1] // 3
    info = {
        "source": os.path.abspath(ply_in),
        "ply": os.path.abspath(ply_out),
        "n": int(gaussians["xyz"].shape[0]),
        "sh_degree": int(round(math.sqrt(n_sh_coeffs))) - 1,
        "baked": True,
        "filter_3D": has_filter_3D,
    }
    if label_counts is not None:
        info["labels"] = label_counts
    return info


def load_3dgs_ply_for_gsplat(path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Activated parameters of a standard 3DGS PLY, laid out as artifixer_gs.render.read_ply returns them."""
    gaussians = read_gw_ply(path)
    n_gaussians = gaussians["xyz"].shape[0]
    n_sh_coeffs = 1 + gaussians["f_rest"].shape[1] // 3
    sh = np.concatenate(
        [
            gaussians["f_dc"][:, None, :],
            gaussians["f_rest"].reshape(n_gaussians, 3, n_sh_coeffs - 1).transpose(0, 2, 1),
        ],
        axis=1,
    )  # (N, K, 3)
    rotation = gaussians["rotation"] / np.linalg.norm(gaussians["rotation"], axis=1, keepdims=True)
    return {
        "means": torch.from_numpy(gaussians["xyz"]).to(device),
        "quats": torch.from_numpy(rotation).to(device),
        "scales": torch.exp(torch.from_numpy(gaussians["scaling"])).to(device),
        "opacities": torch.sigmoid(torch.from_numpy(gaussians["opacity"][:, 0])).to(device),
        "colors": torch.from_numpy(np.ascontiguousarray(sh)).to(device),
        "sh_degree": int(round(math.sqrt(n_sh_coeffs))) - 1,
    }


def camera_matrices(camera):
    """
    gsplat camera of a GW Camera: OpenCV world-to-camera viewmat (4, 4), K (3, 3), width, height (numpy float64).
    GW projects pixel centres at integers (ndc2Pix) and gsplat at +0.5, so the principal point is (W / 2, H / 2).
    """
    W, H = int(camera.image_width), int(camera.image_height)
    focal_x = W / (2. * math.tan(camera.FoVx / 2.))
    focal_y = H / (2. * math.tan(camera.FoVy / 2.))
    K = np.array([[focal_x, 0., W / 2.], [0., focal_y, H / 2.], [0., 0., 1.]], dtype=np.float64)
    viewmat = camera.world_view_transform.detach().double().cpu().numpy().T  # GW stores it transposed
    return viewmat, K, W, H


def native_views(scene, gaussians, pipe, background: torch.Tensor, kernel_size: float, splits=("train", "test"),
                 stride: int = 1):
    """Yield (name, split, camera, rgb (H, W, 3), alpha (H, W), photo (H, W, 3)) of render_ours, numpy float32."""
    from gaussian_renderer.ours import render_ours

    for split in splits:
        cameras = {"train": scene.getTrainCameras, "test": scene.getTestCameras}[split]()
        for i, camera in enumerate(cameras[::stride]):
            render_pkg = render_ours(camera, gaussians, pipe, background, kernel_size=kernel_size)
            rgb = render_pkg["render"].clamp(0., 1.).permute(1, 2, 0).float().cpu().numpy()
            alpha = render_pkg["alpha"][0].float().cpu().numpy()
            gt = camera.original_image[:3].clamp(0., 1.).permute(1, 2, 0).float().cpu().numpy()
            yield f"{split}_{i * stride:04d}", split, camera, rgb, alpha, gt


OURS_PLY_REPORT = "ours_ply_check.json"


def with_ours_ply(views, ply_model, pipe, background: torch.Tensor, kernel_size: float, rows: list):
    """Pass the native views on; render each one also from the exported PLY with render_ours (rows += compare_view).
    This is what ArtiFixer-GS sees through RENDER_CMD=render_ply_frames.py: it must match native (no 3DGS renderer)."""
    from gaussian_renderer.ours import render_ours

    for name, split, camera, rgb, alpha, gt in views:
        out = render_ours(camera, ply_model, pipe, background, kernel_size=kernel_size)
        ply_rgb = out["render"].clamp(0., 1.).permute(1, 2, 0).float().cpu().numpy()
        ply_alpha = out["alpha"][0].float().cpu().numpy()
        rows.append({"name": name, "split": split, "image_name": camera.image_name,
                     **gsplat_check.compare_view(rgb, alpha, ply_rgb, ply_alpha, gt)})
        yield name, split, camera, rgb, alpha, gt


def dump_native_views(views, dump_dir: str, meta: Dict) -> int:
    """Write native_views() output to dump_dir in the gsplat_check.py layout. Returns the number of views."""
    os.makedirs(dump_dir, exist_ok=True)
    entries = []
    for name, split, camera, rgb, alpha, gt in views:
        file = gsplat_check.dump_view(dump_dir, name, rgb, alpha, gt)
        viewmat, K, W, H = camera_matrices(camera)
        entries.append(gsplat_check.view_entry(name, split, getattr(camera, "image_name", name), file, W, H, viewmat, K))
    gsplat_check.write_manifest(dump_dir, meta, entries)
    return len(entries)


def _dump_files(dump_dir: str):
    """Files of dump_dir; raises ValueError if anything is not a gsplat_check dump file (manifest.json, *.npz)."""
    names = sorted(os.listdir(dump_dir))
    other = [n for n in names if not os.path.isfile(os.path.join(dump_dir, n))
             or not (n == gsplat_check.MANIFEST or n.endswith(".npz"))]
    if other:
        raise ValueError(f"{dump_dir} is not a gsplat_check dump (holds {', '.join(other[:5])}); "
                         f"refusing to clear it, pass an empty or new --check_dir")
    return names


def clear_dump(dump_dir: str):
    """Remove a previous dump in dump_dir (only manifest.json + *.npz; anything else: ValueError, nothing removed)."""
    if not os.path.exists(dump_dir):
        return
    if not os.path.isdir(dump_dir):
        raise ValueError(f"--check_dir {dump_dir} is not a directory")
    for name in _dump_files(dump_dir):
        os.remove(os.path.join(dump_dir, name))


def remove_dump(dump_dir: str):
    """
    Remove the dump files (manifest.json, *.npz) of dump_dir, then dump_dir itself if it is left empty.
    check_export clears dump_dir with clear_dump() first, so these are the files this run wrote; others are kept.
    """
    if not os.path.isdir(dump_dir):
        return
    for name in os.listdir(dump_dir):
        path = os.path.join(dump_dir, name)
        if os.path.isfile(path) and (name == gsplat_check.MANIFEST or name.endswith(".npz")):
            os.remove(path)
    if not os.listdir(dump_dir):
        os.rmdir(dump_dir)


# what check_export catches (logged into export_info.json["check"]["error"]) instead of failing the export:
# the worker exiting non-zero or missing (CalledProcessError, OSError), a bad report (ValueError), CUDA OOM (RuntimeError)
CHECK_ERRORS = (subprocess.CalledProcessError, OSError, ValueError, RuntimeError)


def run_gsplat_worker(gs_python: str, dump_dir: str, out_json: str, max_drop: float, extra_args=()) -> Dict:
    """Run gsplat_check.py in gs_python (another env: PYTHONPATH cleared, cwd = repo root) and return its report."""
    dump_dir, out_json = os.path.abspath(dump_dir), os.path.abspath(out_json)
    command = [gs_python, os.path.join(BASE_DIR, "gsplat_check.py"), "--views", dump_dir, "--out", out_json,
               "--max_drop", repr(float(max_drop)), *extra_args]
    print(f"[INFO] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=ROOT_DIR, env={**os.environ, "PYTHONPATH": ""})
    with open(out_json) as f:
        return json.load(f)


@torch.no_grad()
def check_export(dataset, pipe, iteration: int, ply_out: str, stride: int = 1, splits=("train", "test"),
                 gs_python: Optional[str] = None, max_drop: float = gsplat_check.DEFAULT_MAX_DROP,
                 dump_dir: Optional[str] = None, keep_views: bool = False, ours_ply: bool = True) -> Optional[Dict]:
    """
    Render the views natively (render_ours) and with gsplat from the exported PLY; write <export dir>/gsplat_check.json.
    ours_ply: also render the exported PLY with render_ours (GaussianModel.load_ply, no filter_3D: the baked model, as
    render_ply_frames.py renders it for ArtiFixer-GS) -> <export dir>/ours_ply_check.json, key "ours_ply" of the result.
    gsplat in this env: in-process. Otherwise the native views are dumped and gsplat_check.py runs in gs_python.
    Returns the report without its per-view rows, None when no gsplat is available, or {"error": ...} when the check
    fails (CHECK_ERRORS: the export never fails on it). dump_dir must be new, empty or a previous dump (else ValueError,
    raised before any rendering); a relative dump_dir is resolved against the caller's cwd.
    """
    from scene import Scene, GaussianModel

    in_process = gsplat_check.gsplat_available()
    gs_python = gs_python or os.environ.get("GS_RENDER_PY")
    use_gsplat = in_process or bool(gs_python)
    if not use_gsplat:
        print("[WARNING] gsplat is not importable and neither --gs_python nor GS_RENDER_PY is set, skipping the gsplat check.")
        if not ours_ply:
            return None
    export_dir = os.path.dirname(os.path.abspath(ply_out))
    out_json = os.path.join(export_dir, gsplat_check.REPORT)
    ours_json = os.path.join(export_dir, OURS_PLY_REPORT)
    for stale in (out_json, ours_json):
        if os.path.isfile(stale):
            os.remove(stale)  # no stale report if this check fails
    if use_gsplat and not in_process:
        dump_dir = os.path.abspath(dump_dir or os.path.join(export_dir, "gsplat_check_views"))
        clear_dump(dump_dir)  # before the GPU work: refuses a dir that is not a dump

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    background = [1., 1., 1.] if dataset.white_background else [0., 0., 0.]
    device = "cuda" if torch.cuda.is_available() else "cpu"  # render_ours needs cuda; cpu only for the unit tests
    meta = {"ply": os.path.abspath(ply_out), "iteration": iteration, "background": background, "stride": stride,
            **gsplat_check.raster_settings(dataset.kernel_size)}
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    views = native_views(scene, gaussians, pipe, bg, dataset.kernel_size, splits, stride)
    ours_rows, ours_result = [], None
    if ours_ply:
        ply_model = GaussianModel(None)  # SH degree from the PLY
        ply_model.load_ply(ply_out)
        views = with_ours_ply(views, ply_model, pipe, bg, dataset.kernel_size, ours_rows)

    def ours_report():
        if not ours_rows:
            return None
        report = gsplat_check.summarize({**meta, "backend": "ours_ply"}, ours_rows, max_drop)
        gsplat_check.write_report(ours_json, report)
        print(f"[INFO] exported PLY with render_ours: {gsplat_check.format_report(report)}\n[INFO] Saved {ours_json}")
        return {key: value for key, value in report.items() if key != "views"}

    if not use_gsplat:
        for _ in views:
            pass
        return {"ours_ply": ours_report()}
    try:
        if in_process:
            exported = load_3dgs_ply_for_gsplat(ply_out)
            exported["sh"] = exported.pop("colors")
            rows = []
            for name, split, camera, rgb, alpha, gt in views:
                viewmat, K, W, H = camera_matrices(camera)
                gs_rgb, gs_alpha = gsplat_check.gsplat_render(exported, viewmat, K, W, H, background, meta["eps2d"],
                                                              meta["rasterize_mode"], meta["near_plane"])
                rows.append({"name": name, "split": split, "image_name": camera.image_name,
                             **gsplat_check.compare_view(rgb, alpha, gs_rgb, gs_alpha, gt)})
            report = gsplat_check.summarize({**meta, "backend": "in-process"}, rows, max_drop)
            gsplat_check.write_report(out_json, report)
        else:
            n_views = dump_native_views(views, dump_dir, meta)
            print(f"[INFO] Dumped {n_views} native views to {dump_dir}")
            ours_result = ours_report()
            del scene, gaussians, views
            if ours_ply:
                del ply_model
            torch.cuda.empty_cache()  # the worker needs the GPU
            report = run_gsplat_worker(gs_python, dump_dir, out_json, max_drop)
    except CHECK_ERRORS as error:  # worker failure, OOM, full disk: keep the export, report the error
        backend = "in-process" if in_process else "subprocess"
        print(f"[WARNING] gsplat check ({backend}) failed, export kept without it: {error!r}"
              + ("" if in_process or keep_views else " (rerun with --keep_check_views to keep the dump)"), flush=True)
        return {"error": repr(error), "backend": backend, **({} if in_process else {"gs_python": gs_python}),
                **({"ours_ply": ours_result or ours_report()} if ours_ply else {})}
    finally:
        if not in_process and not keep_views:
            remove_dump(dump_dir)

    print(f"[INFO] {gsplat_check.format_report(report)}\n[INFO] Saved {out_json}")
    return {**{key: value for key, value in report.items() if key != "views"},
            **({"ours_ply": ours_result or ours_report()} if ours_ply else {})}


if __name__ == "__main__":
    from arguments import ModelParams, PipelineParams, get_combined_args
    from utils.system_utils import searchForMaxIteration

    parser = ArgumentParser(description="Export a Gaussian Wrapping model as a standard 3DGS PLY")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out", default=None, type=str, help="Output PLY. Default: <model>/../export/point_cloud.ply")
    parser.add_argument("--check", action="store_true",
                        help="Compare gsplat and render_ours on --check_splits (GPU); report <export dir>/gsplat_check.json")
    parser.add_argument("--check_stride", default=1, type=int, help="Every n-th view of each split")
    parser.add_argument("--check_splits", default="train,test", type=str, help="Comma-separated: train, test")
    parser.add_argument("--gs_python", default=os.environ.get("GS_RENDER_PY"), type=str,
                        help="Python with torch + gsplat, used when gsplat is not importable here (default: $GS_RENDER_PY)")
    parser.add_argument("--max_drop", default=gsplat_check.DEFAULT_MAX_DROP, type=float,
                        help="Check passes if mean PSNR vs photos drops by at most this (dB) from native to gsplat")
    parser.add_argument("--check_dir", default=None, type=str,
                        help="Dump dir of the native views: new, empty or an earlier dump (default: <export dir>/gsplat_check_views)")
    parser.add_argument("--keep_check_views", action="store_true", help="Keep the dumped native views")
    parser.add_argument("--split_labels", action="store_true", help="Also write label_<k>.ply per BEA-GS label")
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.model_path, "cfg_args")):
        args = get_combined_args(parser)

    if getattr(args, "use_unbounded_opacity", False):
        raise ValueError("Unbounded opacity is view-dependent and cannot be baked into a 3DGS PLY.")

    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    ply_in = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    ply_out = args.out
    if ply_out is None:
        ply_out = os.path.join(os.path.dirname(os.path.normpath(os.path.abspath(args.model_path))), "export", "point_cloud.ply")

    print(f"[INFO] Exporting {ply_in} to {ply_out}")
    info = export_3dgs_ply(ply_in, ply_out, args.split_labels)
    info["iteration"] = iteration
    print(f"[INFO] Exported {info['n']} Gaussians, SH degree {info['sh_degree']}, 3D filter baked: {info['filter_3D']}")

    info_json = os.path.join(os.path.dirname(os.path.abspath(ply_out)), "export_info.json")
    with open(info_json, "w") as f:  # written before the check too, so a crash in it still leaves the export info
        json.dump(info, f, indent=2)

    if args.check:
        splits = tuple(split for split in args.check_splits.split(",") if split)
        info["check"] = check_export(model.extract(args), pipeline.extract(args), iteration, ply_out, args.check_stride,
                                     splits, args.gs_python, args.max_drop, args.check_dir, args.keep_check_views)
        with open(info_json, "w") as f:
            json.dump(info, f, indent=2)

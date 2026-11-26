"""Check a median-depth rasterizer build against another (stock vs full-bracket): dump, then compare.

    python gaussian_wrapping/full_bracket_check.py dump -m <model> --iteration N --out <file.npz> [--n_views 6]
    python gaussian_wrapping/full_bracket_check.py compare <stock.npz> <new.npz> [--ref <other_new.npz>]

Which build is loaded follows PYTHONPATH (RASTER_PATH in slurm/common.sh); dump records the .so paths.
dump, on --n_views cameras (test + train, evenly spread), saves:
  render_<k>   median depth of render_ours (diff_gaussian_rasterization_ours_beags, renderCUDA)
  sample_<k>   sample_depth_with_ours on --n_points Gaussian centres (diff_gaussian_rasterization_ours, sampleDepthCUDA)
  sinside_<k>  its inside flag
  alpha_<k>    integrate_ours alpha on the same points (evaluateSDFCUDA), ainside_<k> its inside flag
  ms_render    mean ms per render_ours call (geometry on), over --timing_reps passes of the views
compare: where the stock build found a depth, the new one must be bit-identical; prints how many stock holes the new
build fills, and (with --ref) how the filled values agree with another build.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def dump(argv):
    import torch
    from beags.common import load_model, parse_args
    from gaussian_renderer.ours import integrate_ours, render_ours, sample_depth_with_ours
    import diff_gaussian_rasterization_ours as ra
    import diff_gaussian_rasterization_ours_beags as rb

    def add_args(p):
        p.add_argument("--out", required=True)
        p.add_argument("--n_views", type=int, default=6)
        p.add_argument("--n_points", type=int, default=200000)
        p.add_argument("--timing_reps", type=int, default=3)
    sys.argv = [sys.argv[0]] + argv
    dataset, pipe, opt, args = parse_args("full bracket check", add_args)
    with torch.no_grad():
        scene, gaussians = load_model(dataset, args.iteration, with_labels=False)
        cams = list(scene.getTestCameras()) + list(scene.getTrainCameras())
        cams = [cams[i] for i in np.unique(np.linspace(0, len(cams) - 1, min(args.n_views, len(cams))).round().astype(int))]
        g = torch.Generator(device="cpu").manual_seed(0)
        xyz = gaussians.get_xyz
        idx = torch.randperm(xyz.shape[0], generator=g)[:args.n_points].to(xyz.device)
        pts = xyz[idx].contiguous()
        bg = torch.zeros(3, device="cuda")
        out = {"so": json.dumps([os.path.dirname(ra.__file__), os.path.dirname(rb.__file__)]),
               "views": json.dumps([c.image_name for c in cams])}
        for k, cam in enumerate(cams):
            out[f"render_{k}"] = render_ours(cam, gaussians, pipe, bg, require_depth=True)["median_depth"][0].cpu().numpy()
            s = sample_depth_with_ours(pts, cam, gaussians, pipe, 0.0)
            out[f"sample_{k}"] = s["sampled_depth"].reshape(-1).cpu().numpy()
            out[f"sinside_{k}"] = s["inside"].reshape(-1).cpu().numpy()
            a = integrate_ours(pts, cam, gaussians, pipe, 0.0)
            out[f"alpha_{k}"] = a["alpha_integrated"].reshape(-1).cpu().numpy()
            out[f"ainside_{k}"] = a["inside"].reshape(-1).cpu().numpy()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.timing_reps):
            for cam in cams:
                render_ours(cam, gaussians, pipe, bg, require_depth=True)
        torch.cuda.synchronize()
        out["ms_render"] = np.float64((time.time() - t0) * 1000 / (args.timing_reps * len(cams)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[fbcheck] dumped {args.out}: {len(cams)} views, {pts.shape[0]} points, "
          f"{float(out['ms_render']):.1f} ms/render, so {out['so']}")


def compare(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stock")
    ap.add_argument("new")
    ap.add_argument("--ref", default=None, help="another build's dump (e.g. v1) to compare the filled values with")
    a = ap.parse_args(argv)
    S, N = np.load(a.stock), np.load(a.new)
    R = np.load(a.ref) if a.ref else None
    ok = True
    k = 0
    rows = []
    while f"render_{k}" in S:
        r = {}
        for key, ins in (("render", None), ("sample", "sinside"), ("alpha", "ainside")):
            s, n = S[f"{key}_{k}"], N[f"{key}_{k}"]
            if ins is None:
                found_s, found_n = s > 0, n > 0
            else:                                    # per-point outputs; integrate gives several values per point
                found_s, found_n = S[f"{ins}_{k}"].astype(bool), N[f"{ins}_{k}"].astype(bool)
                s, n = s.reshape(found_s.shape[0], -1), n.reshape(found_s.shape[0], -1)
            same = np.array_equal(s[found_s], n[found_s]) and bool(found_n[found_s].all())
            ok &= same
            filled = found_n & ~found_s
            r[key] = {"stock_found": float(found_s.mean()), "new_found": float(found_n.mean()),
                      "identical_where_stock_found": same, "filled": int(filled.sum()),
                      "max_abs_diff_where_stock_found": float(np.abs(s[found_s] - n[found_s]).max()) if found_s.any() else 0.0}
            if R is not None and filled.any():
                rr = R[f"{key}_{k}"].reshape(s.shape)[filled]
                rel = np.abs(n[filled] - rr) / np.maximum(np.abs(rr), 1e-6)
                r[key]["filled_rel_diff_vs_ref_p50_p99"] = [float(np.median(rel)), float(np.percentile(rel, 99))]
        rows.append(r)
        k += 1
    for i, r in enumerate(rows):
        print(f"[fbcheck] view {i}: " + json.dumps(r))
    print(f"[fbcheck] ms/render stock {float(S['ms_render']):.1f}  new {float(N['ms_render']):.1f}"
          + (f"  ref {float(R['ms_render']):.1f}" if R is not None else ""))
    print(f"[fbcheck] {'PASS' if ok else 'FAIL'}: new build bit-identical to stock wherever stock found a depth")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dump", "compare"):
        sys.exit(__doc__)
    sys.exit(dump(sys.argv[2:]) if sys.argv[1] == "dump" else compare(sys.argv[2:]))

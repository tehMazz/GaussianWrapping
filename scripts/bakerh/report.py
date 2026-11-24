"""What one BakerH scene produced, as text for REPORT.txt (run_all.sh) and job_summary.txt (run_scene.sh).

    python scripts/bakerh/report.py --output <output> --method <method> --scene S [--scene S2 ...] [--table]

Per scene (output/<method>/<scene>/...): PSNR / SSIM / LPIPS of the test views (train <iter>, beags <ft_iter>,
results.json), gs_stats.json, export (export_info.json, gsplat_check.json: gsplat vs native PSNR / alpha MAE),
BEA-GS segmentation against the SAM3 masks (beags/metrics_<it>.json: IoU / BIoU / EIoU / BEIoU; BakerH has no GT,
so this is agreement with the input masks, not a held-out score), Gaussians per label (labels.pt, if torch imports),
hollow metrics of both models (hollow_metrics_<it><--hm_suffix>.json, legacy hole keys renamed as
hollow_metrics.load_metrics), face counts of every mesh PLY (PLY header), the label renders, Seed2GS
(output/seed2gs/<scene>/*/seed2gs.json + iou_vs_sam3.json) and ArtiFixer (output/artifixer/<scene>/<variant>:
inputs.json, finished run.sh stages (markers run.sh marked 'stale' listed apart), newest ArtiFixer / ArtiFixer3D+
frames, ArtiFixer3D gs_stats.json and PLY). --table: one line per scene instead.
Standard library only (torch optional); a missing file prints as '-'.
"""
import argparse
import glob
import json
import os

LEGACY_KEYS = {"hole_emptiness": "hole_alpha", "hole_emptiness_8x8": "hole_alpha_8x8",
               "hole_unseen_alpha": "hole_alpha_unseen"}  # as gaussian_wrapping/hollow_metrics.py
HOLLOW_KEYS = ("n_gaussians", "unseen_share", "unseen_mass_share", "free_space_mass", "aabb_mass", "floaters",
               "floater_gaussians", "floater_voxel", "floater_voxel_mode", "fog_share_005", "fog_share_010",
               "opacity_entropy", "hole_views", "hole_alpha", "hole_alpha_8x8", "hole_alpha_unseen", "hole_tightness")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def hollow(path):
    d = load(path)
    if d is None:
        return None
    for old, new in LEGACY_KEYS.items():
        if old in d:
            d.setdefault(new, d.pop(old))
    return d


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{nd}f}"
    return str(v)


def ply_faces(path):
    """(vertices, faces) from a PLY header."""
    counts = {}
    try:
        with open(path, "rb") as f:
            for _ in range(64):
                line = f.readline().decode("ascii", "replace").strip()
                if line.startswith("element "):
                    _, name, n = line.split()[:3]
                    counts[name] = int(n)
                if line == "end_header" or not line:
                    break
    except OSError:
        return None
    return counts.get("vertex"), counts.get("face")


def label_counts(path):
    try:
        import torch
        labels = torch.load(path, map_location="cpu").long()
    except Exception:  # noqa: BLE001  (no torch, no file)
        return None
    per = torch.bincount(labels[labels >= 0]).tolist() if (labels >= 0).any() else []
    return len(labels), int((labels < 0).sum()), per


def newest_pred(root):
    hits = glob.glob(os.path.join(root, "**", "frames", "batch_0000", "pred"), recursive=True)
    if not hits:
        return None
    best = max(hits, key=os.path.getmtime)
    return best, len([f for f in os.listdir(best) if f.endswith((".png", ".jpg"))])


class Scene:
    def __init__(self, a, scene):
        self.a, self.s = a, scene
        self.base = os.path.join(a.output, a.method, scene)
        self.m, self.b = os.path.join(self.base, "train"), os.path.join(self.base, "beags")
        self.af = os.path.join(a.output, "artifixer", scene, a.af_variant)

    def psnr(self, model, it):
        d = load(os.path.join(model, "results.json")) or {}
        return d.get(f"ours_{it}")

    def seg(self, model, it):
        d = load(os.path.join(model, "beags", f"metrics_{it}.json")) or {}
        return {split: d[split] for split in ("test", "train") if split in d}

    def lines(self):
        a, out = self.a, []
        out.append(f"{self.s}: {self.base}")
        for name, model, it in (("train", self.m, a.iter), ("beags", self.b, a.ft_iter)):
            r = self.psnr(model, it)
            out.append(f"  test views {name} {it}: " + (", ".join(f"{k} {v:.4f}" for k, v in r.items()) if r else "-"))
        g = load(os.path.join(self.m, "gs_stats.json"))
        if g:
            out.append("  gs_stats train: " + ", ".join(f"{k} {fmt(v)}" for k, v in g.items()))
        e = load(os.path.join(self.base, "export", "export_info.json"))
        c = load(os.path.join(self.base, "export", "gsplat_check.json"))
        out.append(f"  export: {fmt(e and e.get('n'))} Gaussians; gsplat check: "
                   + (", ".join(f"{k} {fmt(c.get(k))}" for k in ("n_views", "psnr", "alpha_mae", "gt_drop") if k in c)
                      if c else "none (no gsplat)"))
        for name, model, it in (("train", self.m, a.iter), ("beags", self.b, a.ft_iter)):
            lc = label_counts(os.path.join(model, "point_cloud", f"iteration_{it}", "labels.pt"))
            if lc:
                out.append(f"  labels {name} {it}: {lc[0]} Gaussians, never hit {lc[1]}, per label {lc[2]}")
            for split, d in self.seg(model, it).items():
                out.append(f"  BEA-GS vs SAM3 masks {name} {it} {split} ({len(d.get('per_view', []))} rows): "
                           + ", ".join(f"{k} {fmt(d.get(k))}" for k in ("IoU", "BIoU", "EIoU", "BEIoU")))
        for name, model, it in (("train", self.m, a.iter), ("beags", self.b, a.ft_iter)):
            h = hollow(os.path.join(model, f"hollow_metrics_{it}{a.hm_suffix}.json"))
            out.append(f"  hollow {name} {it}: "
                       + (", ".join(f"{k} {fmt(h[k])}" for k in HOLLOW_KEYS if k in h) if h else "-"))
        meshes = sorted(glob.glob(os.path.join(self.base, "mesh", "*.ply")))
        pivot = os.path.join(self.b, "mesh_ours_2pivots.ply")
        if os.path.exists(pivot):
            meshes.insert(0, pivot)
        if meshes:
            out.append("  mesh faces: " + ", ".join(
                f"{os.path.relpath(p, self.base)} {fmt((ply_faces(p) or (None, None))[1])}" for p in meshes))
        else:
            out.append("  mesh: -")
        rl = os.path.join(self.base, f"renders_labels_{a.ft_iter}")
        if os.path.isdir(rl):
            dirs = sorted(d for d in os.listdir(rl) if os.path.isdir(os.path.join(rl, d)))
            n = len(os.listdir(os.path.join(rl, "full"))) if "full" in dirs else 0
            out.append(f"  label renders: {rl} ({n} views: {', '.join(dirs)}"
                       + (", grid_all.jpg" if os.path.exists(os.path.join(rl, "grid_all.jpg")) else "") + ")")
        for info in sorted(glob.glob(os.path.join(a.output, "seed2gs", self.s, "*", "seed2gs.json"))):
            d, iou = load(info) or {}, load(os.path.join(os.path.dirname(info), "iou_vs_sam3.json")) or {}
            out.append(f"  seed2gs '{d.get('prompt')}': SAM3 score {fmt(d.get('sam3_score'))}, object "
                       f"{d.get('n_object')} of {d.get('n_gaussians')} Gaussians, {fmt((d.get('seconds') or {}).get('total'), 1)} s; "
                       f"mask IoU vs SAM3 {fmt(iou.get('mean_iou'))} over {iou.get('views', '-')} views")
        out.extend(self.artifixer())
        return out

    def artifixer(self):
        if not os.path.isdir(self.af):
            return [f"  artifixer {self.a.af_variant}: -"]
        out = []
        d = load(os.path.join(self.af, "inputs.json")) or {}
        m = d.get("mask_stats") or {}
        out.append(f"  artifixer {self.a.af_variant} inputs: object {d.get('object_gaussians')} of {d.get('num_gaussians')} "
                   f"Gaussians, masks {m.get('masks', '-')} of {m.get('photos', '-')} photos (object in "
                   f"{m.get('object_views', '-')}), hole {d.get('HOLE_MODE', '-')}, metric scale {fmt(d.get('metric_scale'))} "
                   f"({d.get('metric_scale_source', '-')}), caption.h5 {d.get('CAPTION_H5', '-')}")
        root = os.path.join(self.af, self.s)
        done, stale = [], []  # run.sh writes 'stale' into the markers of the stages after a rerun one
        for p in sorted(glob.glob(os.path.join(root, "logs", ".done_*"))):
            try:
                with open(p) as f:
                    is_stale = f.read().strip() == "stale"
            except OSError:
                is_stale = True
            (stale if is_stale else done).append(os.path.basename(p)[len(".done_"):])
        out.append(f"  artifixer stages done: {', '.join(done) or '-'}"
                   + (f"; stale (outputs from other settings, rerun): {', '.join(stale)}" if stale else ""))
        for name, sub in (("ArtiFixer", "artifixer"), ("ArtiFixer3D+", "artifixer3d_plus")):
            p = newest_pred(os.path.join(root, sub))
            out.append(f"  {name}: " + (f"{p[0]} ({p[1]} frames)" if p else "-"))
        g = load(os.path.join(root, "af3d", "fastgs", "gs_stats.json"))
        r = load(os.path.join(root, "af3d", "fastgs", "results.json")) or {}
        ply = os.path.join(root, "af3d", "export", "point_cloud.ply")
        out.append(f"  ArtiFixer3D: PLY {ply if os.path.exists(ply) else '-'}; gs_stats "
                   + (", ".join(f"{k} {fmt(v)}" for k, v in g.items()) if g else "-")
                   + ("; " + json.dumps(r) if r else ""))
        return out

    def row(self):
        a = self.a
        p30, p33 = self.psnr(self.m, a.iter) or {}, self.psnr(self.b, a.ft_iter) or {}
        seg = self.seg(self.b, a.ft_iter)
        seg = seg.get("test") or seg.get("train") or {}
        h = hollow(os.path.join(self.b, f"hollow_metrics_{a.ft_iter}{a.hm_suffix}.json")) or {}
        hj = load(os.path.join(self.base, "mesh", f"hole_{a.label}.json")) or {}
        hole_faces = (f"{hj['object_faces']}+{hj['patch_faces']}" if "patch_faces" in hj
                      else fmt((ply_faces(os.path.join(self.base, "mesh", f"hole_{a.label}.ply")) or (None, None))[1]))
        afp = newest_pred(os.path.join(self.af, self.s, "artifixer3d_plus"))
        return (f"{self.s:14s} {fmt(p30.get('PSNR'), 2):>7s} {fmt(p33.get('PSNR'), 2):>7s} {fmt(seg.get('IoU'), 3):>6s} "
                f"{fmt(seg.get('BIoU'), 3):>6s} {fmt(h.get('hole_alpha'), 3):>6s} {fmt(h.get('hole_alpha_unseen'), 3):>8s} "
                f"{fmt(h.get('floaters')):>8s} {hole_faces:>11s} {('%d fr' % afp[1]) if afp else '-':>7s}")


def main():
    p = argparse.ArgumentParser(description="BakerH per-scene report")
    p.add_argument("--output", required=True, help="output root (common.sh OUTPUT)")
    p.add_argument("--method", required=True, help="output/<method>/<scene> (common.sh METHOD)")
    p.add_argument("--scene", action="append", required=True)
    p.add_argument("--iter", type=int, default=30000)
    p.add_argument("--ft_iter", type=int, default=33000)
    p.add_argument("--label", type=int, default=1)
    p.add_argument("--af_variant", default="normal")
    p.add_argument("--hm_suffix", default="", help="hollow_metrics_<it><suffix>.json (common.sh hm_suffix of HM_ARGS)")
    p.add_argument("--table", action="store_true", help="one line per scene")
    a = p.parse_args()
    if a.table:
        print(f"{'scene':14s} {'PSNR30':>7s} {'PSNR33':>7s} {'IoU':>6s} {'BIoU':>6s} {'hole_a':>6s} {'hole_a_u':>8s} "
              f"{'floaters':>8s} {'hole obj+pt':>11s} {'AF3D+':>7s}")
        for s in a.scene:
            print(Scene(a, s).row())
        return
    for s in a.scene:
        print("\n".join(Scene(a, s).lines()))


if __name__ == "__main__":
    main()

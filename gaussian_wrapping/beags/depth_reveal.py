"""Before / after depth study of an object removal: which revealed pixels did the training views ever see?

    python gaussian_wrapping/beags/depth_reveal.py -m <model> [--iteration N] --remove <k> --out <dir>
        [--stride 4] [--tol 0.02] [--reveal_tol 0.02] [--radius 1] [--min_alpha 0.5] [--min_seen 3] [--n_vis 8]

For each studied camera v (every --stride-th of train + test), render the full model ("with", object present) and the
model without label k ("without"), both as GW surface z-depth (median depth, expected depth where GW has no median,
0 where alpha < --min_alpha). Pixel classes in v:
    revealed  both have a surface and |D_without - D_with| > reveal_tol * D_with: removing k changed the surface
    emptied   the full render has a surface, the removal render has none (hollowing left the space behind k empty)
    unchanged everything else
A single view cannot tell "hidden only here" from "never seen" (every revealed pixel changes depth by definition), so
each revealed pixel is back-projected to its 3D surface point X (D_without) and checked in every training view q != v,
against q's FULL-model depth D_with_q (what q actually observed) and q's object mask (label k alone, alpha >= 0.5):
    could see  X projects inside q's image, in front of the camera
    seen       |z_q(X) - D_with_q| <= tol * z_q(X) at some non-object pixel within --radius of the projection
               (an object pixel never counts: the object's own surface can lie within tol of X near its base)
    occluded   not seen and the centre pixel's D_with_q is closer than X; occluded_obj: and that pixel is the object
    behind     not seen and q saw past X, or saw nothing there
Never seen = fewer than --min_seen views saw X. The hole score is about that count, not about the share of views
(a table point behind the object from half the orbit is still well seen from the other half):
    S = clip(1 - seen / min_seen, 0, 1) on revealed pixels, 1 on emptied pixels, 0 elsewhere
share = seen / could_see is kept as a statistic. best_angle: the smallest angle (deg) between v's ray to X and the ray of
a training view that saw X (how far from this direction the nearest observation was).

Writes into --out (never inside the model dir):
    <stem>.npz   uint8 class (0 unchanged, 1 revealed, 2 emptied), float16 S, rel_dd (|dD| / D_with), best_angle,
                 uint16 seen / could / occluded / occluded_obj / behind, uint8 alpha_with / alpha_without / alpha_object
    <stem>.png   255 * S (the unseen-map encoding of beags/unseen_map.py: white = never seen)
    vis/<stem>.jpg  photo | full | removal | object alpha | class // |dD|/D | seen | occluded by object | share | angle
    grid_all.jpg  photo | removal | |dD|/D | never seen over the removal render, one row per vis view
    hist.png      seen count and share on revealed silhouette pixels, dark (removal luminance < 0.2) vs the rest
    study.json    settings, per view and pooled fractions (never seen at seen < 1 / 3 / 5 and per extra --tol in --tols)
"""
import json
import os
import sys

if __name__ == "__main__":  # run as a file: gaussian_wrapping/ on sys.path (not beags/), as the other scripts
    _GW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[0] = _GW
    sys.path.append(os.path.dirname(_GW))
    sys.path.append(os.path.join(os.path.dirname(_GW), "submodules"))

import numpy as np

UNCHANGED, REVEALED, EMPTIED = 0, 1, 2
DARK = 0.2  # removal-render luminance below this = "smear" pixels (HANDOFF 5d)
SEEN_LEVELS = (1, 3, 5)  # study.json: never seen at seen < each


# ------------------------------------------------------------------------------------------ geometry (CPU testable)
class PinholeView:
    """GW camera geometry: row-vector world_view_transform (p_view = [p, 1] @ w2v), pixel (j, i) centre at
    u = fx x / z + cx with cx = (W - 1) / 2 (scene/cameras.py, the rasterizer's ndc2Pix)."""

    def __init__(self, w2v, fx, fy, cx, cy, height, width):
        import torch
        self.w2v = torch.as_tensor(w2v, dtype=torch.float32)
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.height, self.width = int(height), int(width)
        self.rot_inv = torch.linalg.inv(self.w2v[:3, :3])
        self.center = (-self.w2v[3, :3]) @ self.rot_inv

    @classmethod
    def from_camera(cls, cam):
        return cls(cam.world_view_transform.detach(), cam.Fx, cam.Fy, cam.Cx, cam.Cy, cam.image_height, cam.image_width)

    def to(self, device):
        for k in ("w2v", "rot_inv", "center"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def project(self, xyz):
        """(N, 3) world -> u, v (pixel coords, centre of pixel j at u = j), z (camera depth)."""
        import torch
        p = xyz @ self.w2v[:3, :3] + self.w2v[3, :3]
        z = p[:, 2]
        zs = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
        return self.fx * p[:, 0] / zs + self.cx, self.fy * p[:, 1] / zs + self.cy, z

    def backproject(self, rows, cols, z):
        """Pixel indices (rows i, cols j) with z-depth -> (N, 3) world points."""
        import torch
        x = (cols.float() - self.cx) / self.fx * z
        y = (rows.float() - self.cy) / self.fy * z
        return (torch.stack([x, y, z], dim=1) - self.w2v[3, :3]) @ self.rot_inv


def check_views(xyz, ray_dir, refs, ref_depths, tols, radius=1, near=1e-3, skip=None, ref_object=None):
    """Multi-view consistency of world points against full-model depth maps.

    xyz (N, 3); ray_dir (N, 3) unit direction of the studied view's rays (camera -> X); refs [PinholeView];
    ref_depths (Q, H, W) z-depth (0 = no surface); tols: main tolerance first; ref_object (Q, H, W) bool or None:
    pixels of q that show the removed object (never a match). Returns dict of (N,) tensors: could, occluded,
    occluded_obj, behind (int32), seen (len(tols), N) int32, best_cos (N,) max cosine between ray_dir and the ray of a
    view that saw X at tols[0] (-1 when none). occluded / behind also use tols[0]."""
    import torch
    n, dev = xyz.shape[0], xyz.device
    zeros = lambda: torch.zeros(n, dtype=torch.int32, device=dev)
    could, occluded, occluded_obj, behind = zeros(), zeros(), zeros(), zeros()
    seen = torch.zeros(len(tols), n, dtype=torch.int32, device=dev)
    best_cos = torch.full((n,), -1.0, device=dev)
    offs = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    tol_t = torch.tensor(tols, device=dev, dtype=torch.float32).reshape(-1, 1)
    for q, (ref, dmap) in enumerate(zip(refs, ref_depths)):
        if skip is not None and q == skip:
            continue
        u, v, z = ref.project(xyz)
        j, i = torch.round(u).long(), torch.round(v).long()
        inside = (z > near) & (j >= 0) & (j < ref.width) & (i >= 0) & (i < ref.height)
        if not inside.any():
            continue
        idx = inside.nonzero(as_tuple=True)[0]
        zi, ii, jj = z[idx], i[idx], j[idx]
        obj = ref_object[q] if ref_object is not None else None
        best = torch.full_like(zi, float("inf"))  # min relative |z - d| over the window (valid, non-object d only)
        for dy, dx in offs:
            yy, xx = (ii + dy).clamp(0, ref.height - 1), (jj + dx).clamp(0, ref.width - 1)
            d = dmap[yy, xx]
            valid = d > 0 if obj is None else (d > 0) & ~obj[yy, xx]
            best = torch.minimum(best, torch.where(valid, (zi - d).abs() / zi, torch.full_like(zi, float("inf"))))
        ok = best[None, :] <= tol_t                                       # (T, M)
        dc = dmap[ii, jj]
        occ = ~ok[0] & (dc > 0) & (dc < zi * (1 - tols[0]))
        could[idx] += 1
        seen[:, idx] += ok.int()
        occluded[idx] += occ.int()
        if obj is not None:
            occluded_obj[idx] += (occ & obj[ii, jj]).int()
        behind[idx] += (~ok[0] & ~occ).int()
        if ok[0].any():
            k = idx[ok[0]]
            d_q = torch.nn.functional.normalize(xyz[k] - ref.center.reshape(1, 3), dim=1)
            best_cos[k] = torch.maximum(best_cos[k], (d_q * ray_dir[k]).sum(1))
    return {"could": could, "seen": seen, "occluded": occluded, "occluded_obj": occluded_obj, "behind": behind,
            "best_cos": best_cos}


def classify(d_with, a_with, d_without, a_without, reveal_tol, min_alpha):
    """(H, W) uint8 class map (UNCHANGED / REVEALED / EMPTIED) and rel |dD| / D_with (0 where undefined)."""
    import torch
    sw, so = (a_with >= min_alpha) & (d_with > 0), (a_without >= min_alpha) & (d_without > 0)
    rel = torch.where(sw & so, (d_without - d_with).abs() / d_with.clamp_min(1e-8), torch.zeros_like(d_with))
    cls = torch.zeros_like(d_with, dtype=torch.uint8)
    cls[sw & so & (rel > reveal_tol)] = REVEALED
    cls[sw & ~so] = EMPTIED
    return cls, rel


def score(cls, seen, min_seen=3):
    """(H, W) S: clip(1 - seen / min_seen, 0, 1) on revealed pixels, 1 on emptied, 0 elsewhere."""
    import torch
    s = torch.zeros(cls.shape, dtype=torch.float32, device=cls.device)
    rev = cls == REVEALED
    s[rev] = (1 - seen[rev].float() / max(int(min_seen), 1)).clamp(0, 1)
    s[cls == EMPTIED] = 1
    return s


# ------------------------------------------------------------------------------------------------ rendering (GPU)
def surface(cam, gaussians, pipe, mask, min_alpha=0.5):
    """z-depth (H, W) as render_ply_frames.surface_depth (median, else expected; 0 where alpha < min_alpha), alpha, rgb."""
    import torch
    from gaussian_renderer.ours import render_ours
    bg = torch.zeros(3, device="cuda")
    out = render_ours(cam, gaussians, pipe, bg, require_depth=True, mask=mask)
    alpha = out["alpha"][0]
    w2v = cam.world_view_transform
    z = (gaussians.get_xyz @ w2v[:3, 2] + w2v[3, 2])[:, None].repeat(1, 3)
    expected = render_ours(cam, gaussians, pipe, bg, require_depth=False, mask=mask, colors_precomp=z)["render"][0]
    expected = expected / alpha.clamp_min(1e-6)
    median = out["median_depth"][0]
    depth = torch.where(median > 0, median, expected)
    return torch.where(alpha < min_alpha, torch.zeros_like(depth), depth), alpha, out["render"].clamp(0, 1)


def alpha_only(cam, gaussians, pipe, mask):
    import torch
    from gaussian_renderer.ours import render_ours
    return render_ours(cam, gaussians, pipe, torch.zeros(3, device="cuda"), require_depth=False, mask=mask)["alpha"][0]


# ------------------------------------------------------------------------------------------------------ figures
def to8(img):
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)


def class_rgb(cls):
    pal = np.array([[30, 30, 30], [255, 170, 0], [0, 160, 255]], np.uint8)  # unchanged, revealed, emptied
    return pal[cls]


def colormap(x, name, vmin=0.0, vmax=1.0):
    import matplotlib
    t = np.clip((np.nan_to_num(x) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    return np.round(matplotlib.colormaps[name](t)[..., :3] * 255).astype(np.uint8)


def overlay(rgb8, s, cls):
    """S over the removal render on revealed / emptied pixels: red never seen (S 1), blue seen (S 0)."""
    out = rgb8.astype(np.float32)
    m = cls > 0
    tint = np.stack([255 * s, 40 * (1 - s), 255 * (1 - s)], -1)
    out[m] = 0.35 * out[m] + 0.65 * tint[m]
    return out.astype(np.uint8)


def view_figure(path, name, photo, full, removal, a_obj, cls, rel, seen, occ_obj, share, angle, reveal_tol, n_train):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rev = cls == REVEALED
    nan = lambda x: np.where(rev, x, np.nan).astype(np.float32)
    fig, ax = plt.subplots(2, 5, figsize=(25, 7.6))
    panels = [(photo, "photo", None, None), (full, "full render (with object)", None, None),
              (removal, "removal render (without object)", None, None),
              (a_obj, "object alpha (label k alone)", "gray", 1),
              (class_rgb(cls), "class: orange revealed, blue emptied", None, None),
              (nan(rel), f"|D_without - D_with| / D_with (revealed > {reveal_tol})", "magma", 0.5),
              (nan(seen), "seen: training views that observed X (0 = never)", "turbo", 30),
              (nan(occ_obj), "occluded by the object: views where k hid X", "turbo", max(n_train // 4, 1)),
              (nan(share), "share seen = seen / could_see", "RdYlBu", 1),
              (nan(angle), "angle to the nearest seeing view (deg)", "viridis", 90)]
    for a, (img, title, cmap, vmax) in zip(ax.ravel(), panels):
        if cmap is None:
            a.imshow(img)
        else:
            im = a.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            a.set_facecolor("black")
            fig.colorbar(im, ax=a, fraction=0.035, pad=0.01)
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


GRID_NAMES = ["photo", "removal render", "|dD| / D (revealed)", "red never seen (S 1), blue seen (S 0), over removal"]


def grid_row(photo, removal, rel, cls, s, width=360):
    from PIL import Image
    tiles = [to8(photo), to8(removal), colormap(np.where(cls == REVEALED, rel, 0), "magma", 0, 0.5),
             overlay(to8(removal), s, cls)]
    h = round(width * photo.shape[0] / photo.shape[1])
    return np.concatenate([np.asarray(Image.fromarray(t).resize((width, h), Image.BILINEAR)) for t in tiles], axis=1)


def grid_header(width=360):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (len(GRID_NAMES) * width, 22), (255, 255, 255))
    for k, n in enumerate(GRID_NAMES):
        ImageDraw.Draw(im).text((k * width + 4, 5), n, fill=(0, 0, 0))
    return np.asarray(im)


def hist_figure(path, seen, share, occ_obj, dark):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    groups = ((~dark, "revealed, not dark"), (dark, f"revealed, dark (lum < {DARK}): smear"))
    top = int(max(seen.max(), 1)) if len(seen) else 1
    for m, lab in groups:
        if m.any():
            ax[0].hist(np.minimum(seen[m], 40), bins=np.arange(0, 42) - 0.5, alpha=0.6, density=True,
                       label=f"{lab} (n={int(m.sum())})")
            ax[1].hist(share[m], bins=np.linspace(0, 1, 21), alpha=0.6, density=True, label=lab)
            ax[2].hist(occ_obj[m], bins=40, alpha=0.6, density=True, label=lab)
    ax[0].set_xlabel(f"seen count (clipped at 40; max {top})")
    ax[0].set_title("training views that observed the revealed point")
    ax[1].set_xlabel("seen / could_see")
    ax[1].set_title("share of the views that could see it")
    ax[2].set_xlabel("views where the object hid it")
    ax[2].set_title("occluded by the object")
    for a in ax:
        a.set_ylabel("density")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------- stats
def seen_stats(seen, could, dark, prefix=""):
    """Never-seen fractions (seen < SEEN_LEVELS), medians, split dark / not dark. Arrays over revealed pixels."""
    out = {f"{prefix}n": int(len(seen))}
    if not len(seen):
        return out
    share = np.where(could > 0, seen / np.maximum(could, 1), 0)
    for name, m in (("all", np.ones_like(dark)), ("dark", dark), ("not_dark", ~dark)):
        out[f"{prefix}n_{name}"] = int(m.sum())
        if m.any():
            for lv in SEEN_LEVELS:
                out[f"{prefix}never_seen_lt{lv}_{name}"] = float((seen[m] < lv).mean())
            out[f"{prefix}seen_median_{name}"] = float(np.median(seen[m]))
            out[f"{prefix}share_median_{name}"] = float(np.median(share[m]))
    return out


# ---------------------------------------------------------------------------------------------------------- main
def add_args(parser):
    parser.add_argument("--out", required=True, help="output dir (outside the model dir)")
    parser.add_argument("--remove", type=int, required=True, help="label k of the removed object")
    parser.add_argument("--stride", type=int, default=4, help="study every stride-th camera of train + test")
    parser.add_argument("--tol", type=float, default=0.02, help="relative depth tolerance of 'seen in view q'")
    parser.add_argument("--tols", type=float, nargs="*", default=[0.01, 0.05], help="extra tolerances, stats only")
    parser.add_argument("--reveal_tol", type=float, default=0.02, help="relative depth change of a revealed pixel")
    parser.add_argument("--radius", type=int, default=1, help="pixel window radius of the depth comparison in q")
    parser.add_argument("--min_alpha", type=float, default=0.5, help="alpha of a surface pixel")
    parser.add_argument("--min_seen", type=int, default=3, help="S = clip(1 - seen / min_seen): fewer = never seen")
    parser.add_argument("--n_vis", type=int, default=8, help="views with a figure (vis/, grid_all.jpg)")


def main():
    import torch
    from PIL import Image
    from beags.common import load_model, parse_args
    from beags.unseen_map import encode, keep_mask

    dataset, pipe, _, args = parse_args("Before / after depth study of an object removal", add_args)
    out_dir = os.path.abspath(args.out)
    model = os.path.normpath(os.path.abspath(dataset.model_path))
    assert not out_dir.startswith(model + os.sep), f"{out_dir} is inside the model dir"
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    labels = gaussians.labels.reshape(-1)
    keep, obj = keep_mask(labels, args.remove), labels == args.remove
    assert obj.any(), f"no Gaussian has label {args.remove}"
    tols = [args.tol] + [t for t in args.tols if t != args.tol]
    train = list(scene.getTrainCameras())
    cams = train + list(scene.getTestCameras())
    study = cams[:: max(1, args.stride)]
    vis_names = {c.image_name for c in study[:: max(1, len(study) // max(args.n_vis, 1))][: args.n_vis]}
    print(f"[INFO] {len(train)} training depth maps (reference), {len(study)} studied views, label {args.remove}: "
          f"{int(obj.sum())} of {labels.numel()} Gaussians removed", flush=True)

    refs = [PinholeView.from_camera(c).to("cuda") for c in train]
    shape = (len(train), int(train[0].image_height), int(train[0].image_width))
    ref_depths = torch.zeros(shape, device="cuda")
    ref_object = torch.zeros(shape, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        for q, c in enumerate(train):
            d, _, _ = surface(c, gaussians, pipe, None, args.min_alpha)
            assert d.shape == shape[1:], f"{c.image_name}: {tuple(d.shape)} vs {shape[1:]}"
            ref_depths[q] = d
            ref_object[q] = alpha_only(c, gaussians, pipe, obj) >= 0.5
    train_index = {c.image_name: q for q, c in enumerate(train)}
    print(f"[INFO] reference maps done; object covers {100 * ref_object.float().mean().item():.1f}% of train pixels",
          flush=True)

    per_view, rows = {}, []
    pooled = {k: [] for k in ("seen", "could", "occ_obj", "dark", "cls_sil")}
    pooled_tol = {t: [] for t in tols}
    with torch.no_grad():
        for cam in study:
            d_with, a_with, full = surface(cam, gaussians, pipe, None, args.min_alpha)
            d_wo, a_wo, removal = surface(cam, gaussians, pipe, keep, args.min_alpha)
            a_obj = alpha_only(cam, gaussians, pipe, obj)
            cls, rel = classify(d_with, a_with, d_wo, a_wo, args.reveal_tol, args.min_alpha)
            h, w = cls.shape
            rr, cc = (cls == REVEALED).nonzero(as_tuple=True)
            view = PinholeView.from_camera(cam).to("cuda")
            maps = {k: torch.zeros(h, w, dtype=torch.int32, device="cuda")
                    for k in ("could", "occluded", "occluded_obj", "behind")}
            seen_maps = {t: torch.zeros(h, w, dtype=torch.int32, device="cuda") for t in tols}
            angle = torch.full((h, w), float("nan"), device="cuda")
            if rr.numel():
                xyz = view.backproject(rr, cc, d_wo[rr, cc])
                ray = torch.nn.functional.normalize(xyz - view.center.reshape(1, 3), dim=1)
                res = check_views(xyz, ray, refs, ref_depths, tols, args.radius, skip=train_index.get(cam.image_name),
                                  ref_object=ref_object)
                for k in maps:
                    maps[k][rr, cc] = res[k]
                for t_i, t in enumerate(tols):
                    seen_maps[t][rr, cc] = res["seen"][t_i]
                ang = torch.rad2deg(torch.arccos(res["best_cos"].clamp(-1, 1)))
                angle[rr, cc] = torch.where(res["best_cos"] > -1, ang, torch.full_like(ang, float("nan")))
            s = score(cls, seen_maps[args.tol], args.min_seen)

            np_ = lambda t: t.cpu().numpy()
            s_np, cls_np, rel_np = np_(s), np_(cls), np_(rel)
            seen_np, could_np, occ_obj_np = np_(seen_maps[args.tol]), np_(maps["could"]), np_(maps["occluded_obj"])
            share_np = np.where(could_np > 0, seen_np / np.maximum(could_np, 1), 0).astype(np.float32)
            removal_np = np_(removal.permute(1, 2, 0))
            dark = (removal_np @ np.array([0.299, 0.587, 0.114], np.float32)) < DARK
            sil = np_(a_obj) > 0.05
            np.savez_compressed(os.path.join(out_dir, f"{cam.image_name}.npz"), cls=cls_np, S=s_np.astype(np.float16),
                                rel_dd=rel_np.astype(np.float16), best_angle=np_(angle).astype(np.float16),
                                seen=seen_np.astype(np.uint16), **{k: np_(v).astype(np.uint16) for k, v in maps.items()},
                                alpha_with=to8(np_(a_with)), alpha_without=to8(np_(a_wo)), alpha_object=to8(np_(a_obj)))
            Image.fromarray(encode(s_np)).save(os.path.join(out_dir, f"{cam.image_name}.png"))

            rs = (cls_np == REVEALED) & sil
            pooled["seen"].append(seen_np[rs]); pooled["could"].append(could_np[rs])
            pooled["occ_obj"].append(occ_obj_np[rs]); pooled["dark"].append(dark[rs]); pooled["cls_sil"].append(cls_np[sil])
            for t in tols:
                pooled_tol[t].append(np_(seen_maps[t])[rs])
            nsil = max(int(sil.sum()), 1)
            per_view[cam.image_name] = {
                "silhouette": float(sil.mean()), "revealed_in_silhouette": float(rs.sum() / nsil),
                "emptied_in_silhouette": float(((cls_np == EMPTIED) & sil).sum() / nsil),
                "S_gt_0.5_in_silhouette": float(((s_np > 0.5) & sil).sum() / nsil),
                **seen_stats(seen_np[rs], could_np[rs], dark[rs])}
            if cam.image_name in vis_names:
                photo = cam.original_image[:3].permute(1, 2, 0).float().cpu().numpy()
                view_figure(os.path.join(out_dir, "vis", f"{cam.image_name}.jpg"), cam.image_name, photo,
                            np_(full.permute(1, 2, 0)), removal_np, np_(a_obj), cls_np, rel_np, seen_np, occ_obj_np,
                            share_np, np_(angle), args.reveal_tol, len(train))
                rows.append(grid_row(photo, removal_np, rel_np, cls_np, s_np))
            pv = per_view[cam.image_name]
            print(f"[{cam.image_name}] silhouette {100 * pv['silhouette']:.1f}%: revealed "
                  f"{100 * pv['revealed_in_silhouette']:.1f}%, emptied {100 * pv['emptied_in_silhouette']:.1f}%, "
                  f"S>0.5 {100 * pv['S_gt_0.5_in_silhouette']:.1f}%", flush=True)

    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in pooled.items()}
    dark = cat["dark"].astype(bool)
    hist_figure(os.path.join(out_dir, "hist.png"), cat["seen"], np.where(cat["could"] > 0, cat["seen"] /
                np.maximum(cat["could"], 1), 0), cat["occ_obj"], dark)
    if rows:
        Image.fromarray(np.concatenate([grid_header(rows[0].shape[1] // len(GRID_NAMES))] + rows, axis=0)).save(
            os.path.join(out_dir, "grid_all.jpg"), quality=90)
    summary = {"silhouette_pixels": int(len(cat["cls_sil"])),
               "silhouette_class_fraction": {n: float((cat["cls_sil"] == c).mean()) if len(cat["cls_sil"]) else 0.0
                                             for n, c in (("unchanged", UNCHANGED), ("revealed", REVEALED),
                                                          ("emptied", EMPTIED))},
               **seen_stats(cat["seen"], cat["could"], dark)}
    if len(cat["seen"]):
        summary["occluded_by_object_median"] = float(np.median(cat["occ_obj"]))
        summary["could_median"] = float(np.median(cat["could"]))
    for t in tols[1:]:
        summary.update(seen_stats(np.concatenate(pooled_tol[t]), cat["could"], dark, prefix=f"tol{t}_"))
    info = {"model": model, "iteration": scene.loaded_iter, "label": args.remove, "train_views": len(train),
            "studied_views": len(study), "tol": args.tol, "tols": tols, "reveal_tol": args.reveal_tol,
            "radius": args.radius, "min_alpha": args.min_alpha, "min_seen": args.min_seen, "dark_luminance": DARK,
            "silhouette": "label-k-alone alpha > 0.05; pooled stats are over revealed pixels inside it",
            "summary": summary, "views": per_view}
    with open(os.path.join(out_dir, "study.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("[INFO] summary " + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

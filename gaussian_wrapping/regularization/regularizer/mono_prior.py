"""Monocular depth + normal prior (MoGe-2) for weakly conditioned scenes.

Priors come from mono_prior_prepare.py (<dir>/depth/<image>.npy, <dir>/normal/<image>.npy, <dir>/stats.json).
Two losses, both on the rendered (training) view:
  normal: mean(1 - <n_render, n_prior>) where both are valid. Camera space, OpenCV axes, facing the camera: the same
          frame and form as GW's own depth-normal loss (train.py), so it needs no scale.
  depth : mean(|d_median - s * d_prior| / (s * d_prior)), s = the image's scale fitted to the COLMAP points
          (stats.json), so the prior is in scene units.
Gate: COLMAP's median triangulation angle (per 3D point, the widest angle between two cameras that see it). Wide
angles mean multi-view already constrains depth, so with gate "auto" the prior is used only below max_tri_angle_deg
(spinnerf_7: 3.1 deg -> on; mip360 kitchen: 13.7 deg -> off).
Config: configs/mono_prior/<name>.yaml. Enabled by train.py --mono_prior_dir <dir> [--mono_prior_config <name>].
"""
import json
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def median_triangulation_angle(source_path):
    """Median over COLMAP 3D points of the widest viewing-ray angle (degrees). None if sparse/0 is unreadable."""
    from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, qvec2rotmat
    sp = os.path.join(source_path, "sparse", "0")
    try:
        ex = read_extrinsics_binary(os.path.join(sp, "images.bin")) if os.path.exists(os.path.join(sp, "images.bin")) \
            else read_extrinsics_text(os.path.join(sp, "images.txt"))
    except FileNotFoundError:
        return None
    centers = {k: -qvec2rotmat(v.qvec).T @ v.tvec for k, v in ex.items()}
    pb = os.path.join(sp, "points3D.bin")
    if not os.path.exists(pb):
        return None
    angles = []
    with open(pb, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _, x, y, z = struct.unpack("<Qddd", f.read(32))
            f.read(11)
            L = struct.unpack("<Q", f.read(8))[0]
            ids = {i for i in struct.unpack("<" + "ii" * L, f.read(8 * L))[0::2] if i in centers}
            if len(ids) < 2:
                continue
            d = np.stack([centers[i] for i in ids]) - np.array([x, y, z])
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            angles.append(np.degrees(np.arccos(np.clip((d @ d.T).min(), -1, 1))))
    return float(np.median(angles)) if angles else None


def _ramp(it, start, full, end, w):
    """0 before start, linear up to w at full, w until end (-1 = never ends), 0 after."""
    if it < start or (end >= 0 and it >= end):
        return 0.0
    return w if it >= full else w * (it - start) / max(full - start, 1)


class MonoPrior:
    def __init__(self, cfg, depth, normal, scale):
        self.cfg, self.depth, self.normal, self.scale = cfg, depth, normal, scale

    @classmethod
    def from_args(cls, args, scene):
        if not getattr(args, "mono_prior_dir", None):
            return None
        with open(os.path.join(BASE_DIR, "configs", "mono_prior", f"{args.mono_prior_config}.yaml")) as f:
            cfg = yaml.safe_load(f)
        gate = getattr(args, "mono_prior_gate", None) or cfg.get("gate", "auto")
        if gate == "off":
            print("[mono] prior off (gate = off)")
            return None
        if gate == "auto":
            ang = median_triangulation_angle(args.source_path)
            on = ang is not None and ang < cfg["max_tri_angle_deg"]
            print(f"[mono] median triangulation angle {ang} deg, threshold {cfg['max_tri_angle_deg']} -> "
                  f"{'ON' if on else 'off'}")
            if not on:
                return None
        d = args.mono_prior_dir
        with open(os.path.join(d, "stats.json")) as f:
            stats = json.load(f)
        per = {os.path.splitext(r["image"])[0]: r["scale"] for r in stats["per_image"]}
        gscale = stats["summary"]["scale_median"]
        depth, normal, scale = {}, {}, {}
        for cam in scene.getTrainCameras():
            n = cam.image_name
            pd, pn = os.path.join(d, "depth", n + ".npy"), os.path.join(d, "normal", n + ".npy")
            if not (os.path.exists(pd) and os.path.exists(pn)):
                continue
            depth[n] = torch.from_numpy(np.load(pd))                     # (H, W) float16, CPU
            normal[n] = torch.from_numpy(np.load(pn)).permute(2, 0, 1)  # (3, H, W) float16, CPU
            s = per.get(n, gscale)
            scale[n] = s if s == s else gscale                           # NaN -> global
        n_cams = len(scene.getTrainCameras())
        print(f"[mono] priors for {len(depth)}/{n_cams} train views from {d}; config {args.mono_prior_config}: "
              f"{json.dumps(cfg)}")
        if not depth:
            raise FileNotFoundError(f"no mono priors matched the train camera names in {d}")
        return cls(cfg, depth, normal, scale)

    def weights(self, it):
        c = self.cfg
        return (_ramp(it, c["normal_from_iter"], c["normal_full_iter"], c["normal_until_iter"], c["normal_weight"]),
                _ramp(it, c["depth_from_iter"], c["depth_full_iter"], c["depth_until_iter"], c["depth_weight"]))

    def active(self, it):
        return any(w > 0 for w in self.weights(it))

    def loss(self, cam, render_pkg, it):
        """Weighted prior loss for this view (0-dim tensor) and a dict of unweighted terms for logging."""
        wn, wd = self.weights(it)
        d_med = render_pkg["median_depth"]                                 # (1, H, W)
        dev = d_med.device
        zero = torch.zeros((), device=dev)
        if cam.image_name not in self.depth or (wn <= 0 and wd <= 0):
            return zero, {}
        H, W = d_med.shape[-2:]
        dp = self.depth[cam.image_name].to(dev, torch.float32)[None, None]
        np_ = self.normal[cam.image_name].to(dev, torch.float32)[None]
        if dp.shape[-2:] != (H, W):
            dp = F.interpolate(dp, (H, W), mode="nearest")
            np_ = F.interpolate(np_, (H, W), mode="bilinear", align_corners=False)
        dp, np_ = dp[0] * self.scale[cam.image_name], np_[0]
        nlen = np_.norm(dim=0, keepdim=True)
        np_ = np_ / nlen.clamp_min(1e-6)
        has = d_med > 0
        out, terms = zero, {}
        if wn > 0:
            m = (has & (nlen > 0.5))[0]
            if m.any():
                err = 1.0 - (render_pkg["normal"] * np_).sum(0)
                ln = err[m].mean()
                out = out + wn * ln
                terms["mono_normal"] = float(ln)
        if wd > 0:
            m = (has & (dp > 0))[0]
            if m.any():
                dm, pm = d_med[0][m], dp[0][m]          # index first: dp == 0 off the mask would give NaN grads
                ld = ((dm - pm).abs() / pm).mean()
                out = out + wd * ld
                terms["mono_depth"] = float(ld)
        return out, terms

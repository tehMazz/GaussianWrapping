"""A smooth camera path through a captured scene, shared by every renderer (GW, PGSR, meshes). CPU only.

    python gaussian_wrapping/make_camera_path.py -m <GW model dir> --mode forward|orbit --out <path.json>
           [--frames 240] [--loops 1] [--depth D] [--scale 0.7] [--elev_amp 6]
    python gaussian_wrapping/make_camera_path.py -m <GW model dir> --mode arc --out <path.json>
           (--key_index I [--key_path <path.json>] | --key_name <img>) [--pivot_depth D]
           [--az_deg 25] [--el_deg 0] [--radius_scale 1] [--frames 81] [--width W --height H]
    python gaussian_wrapping/make_camera_path.py -m <GW model dir> --mode probe --out <path.json>
           [--frames 32] [--seed 0] [--probe_radius 1.2 1.5] [--probe_el 10 30] --width W --height H

Reads <model>/cameras.json (all cameras, COLMAP world frame, OpenCV axes: x right, y down, z forward) and
<model>/input.ply (SfM points, for the look-at depth). Both GW and PGSR load COLMAP the same way (no
re-normalisation), so one path drives both.
  forward  (LLFF / SPIn-NeRF): an ellipse in the mean camera plane, radii = --scale x the 70th percentile of the
           camera offsets along the mean right / down axes (stays inside the capture), looking at the point
           --depth (default: median SfM depth along the mean view axis) in front of the mean camera.
  orbit    (Mip-NeRF 360): a circle around the focus (least-squares meeting point of the optical axes), at the
           median camera distance and elevation, the elevation swinging by +-elev_amp degrees.
  arc      (VSE, docs/vse/VSE_PLAN.md 1b): a rigid rotation of a key camera about a pivot on its centre ray, at
           --pivot_depth along the optical axis (the VSE driver passes the GW median depth of the centre pixel;
           default: median SfM depth of the points near the image centre, else the depth of the orbit focus).
           Frame k (t = k / (frames-1)) rotates the key by el_deg*t about the horizontal axis (v x up, v = centre -
           pivot; + raises the camera), then by az_deg*t about the scene up (right-handed, + = counter-clockwise
           seen from above), and scales the pivot distance by 1 + (radius_scale-1)*t. Azimuth, elevation and radius
           change linearly (constant step). Frame 0 is the key w2c verbatim, with the key's FoV, so its roll and
           intrinsics are the key's. The values are reached at the LAST frame: for VidSplat's 25% tail (keep
           frames 0-64 of 81) pass 1.25 x the kept angle. Key: frame --key_index of --key_path (FoV and size from
           that path), or camera --key_index / --key_name of cameras.json (sorted by img_name, as above; FoV from
           that camera's fx, fy).
  probe    (VSE far "probe" views for the geometry teacher; 360 scenes): --frames cameras looking at the orbit
           focus, at radius = median camera distance x U(probe_radius) and elevation = median capture elevation +
           U(probe_el) degrees, azimuth stratified over 360 deg. Deterministic in --seed. Odd frames carry
           "holdout": true (16 used + 16 held out for 32 probes).
path.json: {"width", "height" (render size, = --width/--height), "fovx", "fovy", "frames": [{"w2c": 4x4 OpenCV world->camera}, ...], ...}
For a GW/3DGS Camera: R = w2c[:3, :3].T, T = w2c[:3, 3]. arc/probe frames carry extra per-frame keys (az_deg, ...),
which render_path_gw.py ignores.
"""
import argparse
import json
import math
import os

import numpy as np
from plyfile import PlyData


def look_at_w2c(eye, target, up):
    """OpenCV world->camera: z towards target, y down (against `up`), x = right."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)              # camera->world rotation (columns = camera axes in world)
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ eye
    return w2c


def focus_point(centers, dirs):
    """Least-squares point closest to all optical axes."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def rotation_about(axis, angle):
    """3x3 rotation by `angle` (radians) about the unit `axis` (Rodrigues, right-handed)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def w2c_from_c2w(Rc2w, C):
    w2c = np.eye(4)
    w2c[:3, :3] = Rc2w.T
    w2c[:3, 3] = -Rc2w.T @ C
    return w2c


def arc_frames(w2c_key, pivot_depth, up, az_deg, el_deg=0.0, radius_scale=1.0, frames=81, exact_key=True):
    """Arc of `frames` w2c (4x4) about the pivot at `pivot_depth` on the key camera's centre ray.

    Returns (list of w2c, pivot (3,), list of per-frame dicts). Frame 0 is w2c_key itself when exact_key (the formula
    gives the same pose up to float round-off). The pivot stays on every frame's centre ray."""
    w2c_key = np.asarray(w2c_key, dtype=np.float64)
    R = w2c_key[:3, :3].T                                              # camera->world rotation
    C = -R @ w2c_key[:3, 3]
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    pivot = C + pivot_depth * R[:, 2]
    v = C - pivot
    a_el = np.cross(v, up)
    if np.linalg.norm(a_el) < 1e-9 * max(np.linalg.norm(v), 1e-12):
        raise ValueError("arc: the key looks along the up axis, elevation is undefined")
    a_el /= np.linalg.norm(a_el)
    out, meta = [], []
    for k in range(frames):
        t = k / (frames - 1) if frames > 1 else 0.0
        Rk = rotation_about(up, math.radians(az_deg * t)) @ rotation_about(a_el, math.radians(el_deg * t))
        s = 1.0 + (radius_scale - 1.0) * t
        out.append(w2c_from_c2w(Rk @ R, pivot + s * (Rk @ v)))
        meta.append({"t": t, "az_deg": az_deg * t, "el_deg": el_deg * t, "radius_scale": s})
    if exact_key and frames > 0:
        out[0] = w2c_key.copy()
    return out, pivot, meta


def probe_frames(focus, radius, elev0_deg, up, n, seed=0, radius_range=(1.2, 1.5), el_range=(10.0, 30.0), az0=0.0):
    """n look-at-focus cameras: radius x U(radius_range), elevation elev0 + U(el_range) deg (capped at 85),
    azimuth stratified over 360 deg from az0 (radians), one numpy Generator(seed) for all draws."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    rng = np.random.default_rng(seed)
    out, meta = [], []
    for k in range(n):
        az = az0 + 2 * math.pi * (k + rng.uniform()) / n
        el = min(elev0_deg + rng.uniform(*el_range), 85.0)
        rs = rng.uniform(*radius_range)
        e = math.radians(el)
        eye = focus + radius * rs * (math.cos(e) * (math.cos(az) * e1 + math.sin(az) * e2) + math.sin(e) * up)
        out.append(look_at_w2c(eye, np.asarray(focus, dtype=np.float64), up))
        meta.append({"az_deg": math.degrees(az), "el_deg": el, "radius_scale": rs, "holdout": bool(k % 2)})
    return out, meta


def pivot_depth_from_points(w2c, fovx, fovy, pts, frac=0.1, min_points=20):
    """Median camera-z of the SfM points that project within +-frac of the half-FoV around the image centre."""
    if pts is None:
        return None
    p = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    ok = z > 1e-6
    x, y = p[ok, 0] / z[ok], p[ok, 1] / z[ok]
    near = (np.abs(x) < frac * math.tan(fovx / 2)) & (np.abs(y) < frac * math.tan(fovy / 2))
    if near.sum() < min_points:
        return None
    return float(np.median(z[ok][near]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--mode", choices=["forward", "orbit", "arc", "probe"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=None, help="default: 240 (forward, orbit), 81 (arc), 32 (probe)")
    ap.add_argument("--loops", type=float, default=1.0)
    ap.add_argument("--depth", type=float, default=0.0, help="forward: look-at depth (0 = median SfM depth)")
    ap.add_argument("--scale", type=float, default=0.7, help="forward: ellipse radii x 70th pct camera offset")
    ap.add_argument("--elev_amp", type=float, default=6.0, help="orbit: elevation swing (degrees)")
    ap.add_argument("--width", type=int, default=0,
                    help="render width (the training image width); required except for arc with --key_path")
    ap.add_argument("--height", type=int, default=0)
    # arc
    ap.add_argument("--key_path", default="", help="arc: path.json holding the key frame (default: cameras.json)")
    ap.add_argument("--key_index", type=int, default=-1, help="arc: key frame index in --key_path or cameras.json")
    ap.add_argument("--key_name", default="", help="arc: key camera img_name in cameras.json")
    ap.add_argument("--pivot_depth", type=float, default=0.0,
                    help="arc: pivot z-depth on the key centre ray (0 = SfM median near the centre)")
    ap.add_argument("--az_deg", type=float, default=25.0, help="arc: azimuth at the last frame (degrees)")
    ap.add_argument("--el_deg", type=float, default=0.0, help="arc: elevation change at the last frame (degrees)")
    ap.add_argument("--radius_scale", type=float, default=1.0, help="arc: pivot distance factor at the last frame")
    # probe
    ap.add_argument("--seed", type=int, default=0, help="probe: random seed")
    ap.add_argument("--probe_radius", type=float, nargs=2, default=[1.2, 1.5], help="probe: radius factor range")
    ap.add_argument("--probe_el", type=float, nargs=2, default=[10.0, 30.0],
                    help="probe: elevation offset range over the median capture elevation (degrees)")
    a = ap.parse_args()
    if a.frames is None:
        a.frames = {"forward": 240, "orbit": 240, "arc": 81, "probe": 32}[a.mode]
    if (not a.width or not a.height) and not (a.mode == "arc" and a.key_path):
        ap.error("--width and --height are required")
    if a.mode == "arc":
        return main_arc(a, ap)

    cams = sorted(json.load(open(os.path.join(a.model, "cameras.json"))), key=lambda c: c["img_name"])
    C = np.array([c["position"] for c in cams])                        # camera centres
    Rc2w = np.array([c["rotation"] for c in cams])                     # camera->world rotations
    ref = cams[len(cams) // 2]
    # the model renders at the training image size; cameras.json stores the COLMAP size, FoV is the same
    W0, H0 = ref["width"], ref["height"]
    fovx = 2 * math.atan(W0 / (2 * ref["fx"]))
    fovy = 2 * math.atan(H0 / (2 * ref["fy"]))
    up = -Rc2w[:, :, 1].mean(0)                                        # camera y points down
    up /= np.linalg.norm(up)
    pts = None
    ply = os.path.join(a.model, "input.ply")
    if os.path.exists(ply):
        v = PlyData.read(ply)["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], 1)

    frames = []
    t = np.linspace(0, 2 * math.pi * a.loops, a.frames, endpoint=False)
    if a.mode == "forward":
        c0 = C.mean(0)
        x = Rc2w[:, :, 0].mean(0); x /= np.linalg.norm(x)
        z = Rc2w[:, :, 2].mean(0); z -= x * (z @ x); z /= np.linalg.norm(z)
        y = np.cross(z, x)
        off = C - c0
        ra = a.scale * np.percentile(np.abs(off @ x), 70)
        rb = a.scale * np.percentile(np.abs(off @ y), 70)
        depth = a.depth
        if depth <= 0 and pts is not None:
            d = (pts - c0) @ z
            depth = float(np.median(d[d > 0]))
        target = c0 + depth * z
        for th in t:
            eye = c0 + ra * math.cos(th) * x + rb * math.sin(th) * y
            frames.append(look_at_w2c(eye, target, up))
        info = {"center": c0.tolist(), "target": target.tolist(), "radii": [ra, rb], "depth": depth}
    elif a.mode == "probe":
        dirs = Rc2w[:, :, 2]
        focus = focus_point(C, dirs)
        rel = C - focus
        dist = np.linalg.norm(rel, axis=1)
        elev = np.degrees(np.arcsin(np.clip((rel @ up) / dist, -1, 1)))
        r, e0 = float(np.median(dist)), float(np.median(elev))
        frames, meta = probe_frames(focus, r, e0, up, a.frames, seed=a.seed, radius_range=tuple(a.probe_radius),
                                    el_range=tuple(a.probe_el))
        info = {"focus": focus.tolist(), "radius": r, "elevation_deg": e0, "seed": a.seed,
                "probe_radius": list(a.probe_radius), "probe_el": list(a.probe_el)}
    else:
        dirs = Rc2w[:, :, 2]
        focus = focus_point(C, dirs)
        rel = C - focus
        dist = np.linalg.norm(rel, axis=1)
        elev = np.degrees(np.arcsin(np.clip((rel @ up) / dist, -1, 1)))
        r, e0 = float(np.median(dist)), float(np.median(elev))
        e1 = np.cross(up, np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        az0 = math.atan2(rel[0] @ e2, rel[0] @ e1)                     # start next to the first photo
        for k, th in enumerate(t):
            el = math.radians(e0 + a.elev_amp * math.sin(2 * th))
            az = az0 + th
            eye = focus + r * (math.cos(el) * (math.cos(az) * e1 + math.sin(az) * e2) + math.sin(el) * up)
            frames.append(look_at_w2c(eye, focus, up))
        info = {"focus": focus.tolist(), "radius": r, "elevation_deg": e0}

    # sanity: the path's mean view direction vs the captured cameras'
    vdir = np.array([np.linalg.inv(f)[:3, 2] for f in frames]).mean(0)
    cdir = Rc2w[:, :, 2].mean(0)
    ang = math.degrees(math.acos(np.clip(vdir @ cdir / np.linalg.norm(vdir) / np.linalg.norm(cdir), -1, 1)))
    out = {"mode": a.mode, "source_model": os.path.abspath(a.model), "width": a.width, "height": a.height,
           "colmap_width": W0, "colmap_height": H0,
           "fovx": fovx, "fovy": fovy, "up": up.tolist(), **info, "mean_dir_angle_to_capture_deg": ang,
           "frames": [{"w2c": f.tolist()} for f in frames]}
    if a.mode == "probe":
        for fr, m in zip(out["frames"], meta):
            fr.update(m)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"[path] {a.mode}: {len(frames)} frames, fov {math.degrees(fovx):.1f}x{math.degrees(fovy):.1f} deg, "
          f"{json.dumps(info)}, mean view dir {ang:.1f} deg from the capture's -> {a.out}")


def main_arc(a, ap):
    """--mode arc: see the module docstring."""
    cams = sorted(json.load(open(os.path.join(a.model, "cameras.json"))), key=lambda c: c["img_name"])
    Rc2w = np.array([c["rotation"] for c in cams])
    up = -Rc2w[:, :, 1].mean(0)                                        # same scene up as forward/orbit
    up /= np.linalg.norm(up)
    if a.key_path:
        P = json.load(open(a.key_path))
        if not 0 <= a.key_index < len(P["frames"]):
            ap.error(f"--key_index must be in [0, {len(P['frames'])}) for {a.key_path}")
        w2c_key = np.array(P["frames"][a.key_index]["w2c"], dtype=np.float64)
        fovx, fovy = P["fovx"], P["fovy"]
        W, H = a.width or P["width"], a.height or P["height"]
        W0, H0 = P.get("colmap_width", W), P.get("colmap_height", H)
        key = {"path": os.path.abspath(a.key_path), "index": a.key_index}
    else:
        if a.key_name:
            hits = [i for i, c in enumerate(cams) if c["img_name"] == a.key_name]
            if not hits:
                ap.error(f"no camera named {a.key_name} in cameras.json")
            idx = hits[0]
        elif 0 <= a.key_index < len(cams):
            idx = a.key_index
        else:
            ap.error(f"give --key_name or --key_index in [0, {len(cams)})")
        c = cams[idx]
        w2c_key = w2c_from_c2w(np.array(c["rotation"], dtype=np.float64), np.array(c["position"], dtype=np.float64))
        W0, H0 = c["width"], c["height"]
        fovx = 2 * math.atan(W0 / (2 * c["fx"]))
        fovy = 2 * math.atan(H0 / (2 * c["fy"]))
        W, H = a.width, a.height
        key = {"cameras_json_index": idx, "img_name": c["img_name"]}
    if abs((W / H) / (math.tan(fovx / 2) / math.tan(fovy / 2)) - 1) > 0.01:
        print(f"[path] WARNING: render aspect {W}x{H} differs from the key FoV aspect by > 1%")

    depth, source = a.pivot_depth, "--pivot_depth"
    if depth <= 0:
        pts = None
        ply = os.path.join(a.model, "input.ply")
        if os.path.exists(ply):
            v = PlyData.read(ply)["vertex"]
            pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        depth, source = pivot_depth_from_points(w2c_key, fovx, fovy, pts), "sfm_centre_median"
        if depth is None:
            C = np.array([c["position"] for c in cams])
            focus = focus_point(C, Rc2w[:, :, 2])
            depth = float((focus - (-w2c_key[:3, :3].T @ w2c_key[:3, 3])) @ w2c_key[2, :3])
            source = "orbit_focus"
        if depth <= 0:
            ap.error("could not find a pivot in front of the key camera: pass --pivot_depth")
    frames, pivot, meta = arc_frames(w2c_key, depth, up, a.az_deg, a.el_deg, a.radius_scale, a.frames)
    info = {"key": key, "pivot": pivot.tolist(), "pivot_depth": depth, "pivot_source": source,
            "az_deg": a.az_deg, "el_deg": a.el_deg, "radius_scale": a.radius_scale,
            "az_step_deg": a.az_deg / max(a.frames - 1, 1)}
    out = {"mode": "arc", "source_model": os.path.abspath(a.model), "width": W, "height": H,
           "colmap_width": W0, "colmap_height": H0, "fovx": fovx, "fovy": fovy, "up": up.tolist(), **info,
           "frames": [{"w2c": f.tolist(), **m} for f, m in zip(frames, meta)]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"[path] arc: {len(frames)} frames, fov {math.degrees(fovx):.1f}x{math.degrees(fovy):.1f} deg, "
          f"{json.dumps(info)} -> {a.out}")


if __name__ == "__main__":
    main()

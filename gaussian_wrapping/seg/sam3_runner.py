"""SAM3 worker for seed2gs.py. Runs in a Python that has SAM3's dependencies (the SAM3 repo goes on sys.path),
so the GW process never imports SAM3. One JSON line per answer on stdout, one JSON request per stdin line:

  first line, unrequested, once the model is on the GPU:
      -> {"ready": true, "seconds": {"import", "build", "total"}, "fast_init": bool, "triton_cache": <dir>}
  {"op": "detect", "image": <png/jpg>, "prompt": <text>, "out": <npy>}
      text-prompted detection; out = (H, W) bool mask of the highest-scoring instance.
      -> {"found": bool, "score": float, "scores": [...], "seconds": {"session", "prompt", "total"}}
  {"op": "track", "frames": <dir of 0.jpg, 1.jpg, ...>, "mask": <npy>, "out": <dir>}
      SAM3 tracker from the (H, W) mask on frame 0 (any resolution: SAM3 resizes it), forward; out/<i>.npy (h, w)
      bool per frame, at the frame resolution.
      -> {"frames": n, "nonempty": k, "seconds": {"init_state", "add_mask", "propagate", "total"}}
  {"op": "quit"} -> {"ok": true}
Errors answer {"error": <message>} and the worker keeps serving.

Startup cost (fern smoke, 2026-10-28: ~110 s to ready, then 42 s for the first detect; tracking 48 frames ~5 s):
  import   torch + timm + SAM3 from the shared GPFS venv; cold imports take minutes on a busy node, nothing to do here
           except starting the worker first (seed2gs.py spawns it before importing torch itself);
  build    model construction + 3.3 GB checkpoint. --prefetch reads the checkpoint into the page cache in a thread
           while Python imports; --fast_init skips torch.nn.init during construction (every parameter and persistent
           buffer is then overwritten by the strict checkpoint load, so the weights are identical);
  1st call Triton JIT of SAM3's connected-components kernels, cached in TRITON_CACHE_DIR for later runs.
--fake answers every op without SAM3 (box mask, mask copied to every frame): protocol tests on the CPU only.
"""
import argparse
import contextlib
import json
import os
import sys
import sysconfig
import threading
import time
import traceback

import numpy as np

# Triton (SAM3 post-processing) JIT-compiles a C stub that needs Python.h; uv venvs on the system python have none.
if not os.path.isfile(os.path.join(sysconfig.get_path("include"), "Python.h")) and os.environ.get("SAM3_PY_INCLUDE"):
    os.environ["C_INCLUDE_PATH"] = os.pathsep.join(filter(None, [os.environ["SAM3_PY_INCLUDE"],
                                                                 os.environ.get("C_INCLUDE_PATH")]))

INIT_FNS = ("uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_", "eye_", "dirac_",
            "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_")


@contextlib.contextmanager
def skip_init(enabled=True):
    """torch.nn.init.* become no-ops (modules keep torch.empty values). Only for a model whose weights a strict
    checkpoint load overwrites right after construction."""
    import torch
    saved = {n: getattr(torch.nn.init, n) for n in INIT_FNS if hasattr(torch.nn.init, n)}
    if enabled:
        for n in saved:
            setattr(torch.nn.init, n, lambda tensor, *a, **k: tensor)
    try:
        yield
    finally:
        for n, fn in saved.items():
            setattr(torch.nn.init, n, fn)


def prefetch(path, chunk=64 << 20):
    """Reads `path` in a daemon thread (page cache warm-up while the main thread imports)."""
    def run():
        try:
            with open(path, "rb", buffering=0) as f:
                while f.read(chunk):
                    pass
        except OSError:
            pass
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def detect(predictor, image, prompt, out):
    import torch
    t0 = time.time()
    session = predictor.handle_request(dict(type="start_session", resource_path=image))["session_id"]
    t1 = time.time()
    try:
        res = predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=0, text=prompt))["outputs"]
    finally:
        predictor.handle_request(dict(type="close_session", session_id=session))
    seconds = dict(session=t1 - t0, prompt=time.time() - t1, total=time.time() - t0)
    scores = np.asarray(res["out_probs"], dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return {"found": False, "score": 0.0, "scores": [], "seconds": seconds}
    best = int(scores.argmax())
    mask = res["out_binary_masks"][best]
    mask = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    np.save(out, mask.astype(bool).reshape(mask.shape[-2:]))
    return {"found": True, "score": float(scores[best]), "scores": scores.tolist(), "seconds": seconds}


def track(tracker, frames, mask, out):
    import torch
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = tracker.init_state(video_path=frames)
        n = state["num_frames"]
        t1 = time.time()
        tracker.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(np.load(mask)).bool())
        t2 = time.time()
        nonempty = 0
        for idx, _, _, video_res_masks, _ in tracker.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False, propagate_preflight=True):
            m = (video_res_masks[0, 0] > 0).cpu().numpy()
            nonempty += bool(m.any())
            np.save(os.path.join(out, f"{idx}.npy"), m)
    del state
    torch.cuda.empty_cache()
    seconds = dict(init_state=t1 - t0, add_mask=t2 - t1, propagate=time.time() - t2, total=time.time() - t0)
    return {"frames": int(n), "nonempty": int(nonempty), "seconds": seconds}


def _frame_ids(frames):
    return sorted(int(os.path.splitext(f)[0]) for f in os.listdir(frames)
                  if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg") and os.path.splitext(f)[0].isdigit())


def fake_detect(image, prompt, out):
    """Centre box covering the middle half of the image; no instance when the prompt is 'nothing'."""
    from PIL import Image
    t0 = time.time()
    if prompt == "nothing":
        return {"found": False, "score": 0.0, "scores": [], "seconds": dict(total=time.time() - t0)}
    W, H = Image.open(image).size
    mask = np.zeros((H, W), dtype=bool)
    mask[H // 4:H - H // 4, W // 4:W - W // 4] = True
    np.save(out, mask)
    return {"found": True, "score": 0.9, "scores": [0.9, 0.2], "seconds": dict(total=time.time() - t0)}


def fake_track(frames, mask, out):
    """The frame-0 mask, nearest-resized to each frame, on every frame."""
    from PIL import Image
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    ids = _frame_ids(frames)
    if not ids:
        raise FileNotFoundError(f"no frames in {frames}")
    m0 = np.load(mask)
    for i in ids:
        W, H = Image.open(os.path.join(frames, f"{i}.jpg")).size
        ys = np.arange(H) * m0.shape[0] // H
        xs = np.arange(W) * m0.shape[1] // W
        np.save(os.path.join(out, f"{i}.npy"), m0[ys][:, xs])
    return {"frames": len(ids), "nonempty": len(ids) if m0.any() else 0, "seconds": dict(total=time.time() - t0)}


def serve(lines, write, handlers):
    """Request loop: one JSON request per line -> write(answer dict). Returns on quit or end of input."""
    for line in lines:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            op = req.pop("op")
            if op == "quit":
                write({"ok": True})
                return
            if op not in handlers:
                raise ValueError(f"unknown op {op}")
            res = handlers[op](**req)
        except Exception as e:
            traceback.print_exc()
            res = {"error": f"{type(e).__name__}: {e}"}
        write(res)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="", help="SAM3 repository (put on sys.path)")
    parser.add_argument("--checkpoint", default="", help="sam3.pt")
    parser.add_argument("--no_fast_init", action="store_true", help="run torch.nn.init during model construction")
    parser.add_argument("--no_prefetch", action="store_true", help="do not pre-read the checkpoint while importing")
    parser.add_argument("--fake", action="store_true", help="no SAM3: fixed masks (protocol tests)")
    args = parser.parse_args()
    t0 = time.time()

    # stdout carries the protocol only: library prints go to stderr
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def write(obj):
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    if args.fake:
        write({"ready": True, "seconds": {"import": 0.0, "build": 0.0, "total": time.time() - t0}, "fake": True})
        serve(sys.stdin, write, dict(detect=fake_detect, track=fake_track))
        return

    if not args.no_prefetch and os.path.isfile(args.checkpoint):
        prefetch(args.checkpoint)
    sys.path.insert(0, args.repo)
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    t1 = time.time()
    with skip_init(not args.no_fast_init):
        predictor = Sam3VideoPredictor(checkpoint_path=args.checkpoint)  # strict state-dict load
    tracker = predictor.model.tracker
    tracker.backbone = predictor.model.detector.backbone
    t2 = time.time()
    seconds = {"import": t1 - t0, "build": t2 - t1, "total": t2 - t0}
    print(f"[sam3_runner] ready: {json.dumps({k: round(v, 1) for k, v in seconds.items()})}", file=sys.stderr)
    write({"ready": True, "seconds": seconds, "fast_init": not args.no_fast_init,
           "triton_cache": os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))})
    serve(sys.stdin, write, dict(detect=lambda image, prompt, out: detect(predictor, image, prompt, out),
                                 track=lambda frames, mask, out: track(tracker, frames, mask, out)))


if __name__ == "__main__":
    main()

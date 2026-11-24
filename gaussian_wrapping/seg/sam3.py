"""Client of seg/sam3_runner.py: one SAM3 worker process (its own Python env), model loaded once.

Imports only the standard library, so seed2gs.py can spawn the worker before it imports torch (the worker's imports
and checkpoint load then overlap with the GW side's). The worker announces {"ready": ...} once loaded; the first
call (or wait_ready()) blocks on it and keeps it in .ready, with the client-side wait in .ready["waited"].
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_runner.py")
SAM3_REPO = os.environ.get("SAM3_REPO", "/leonardo_work/IscrC_EditGS/repos/sam3")
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", os.path.join(SAM3_REPO, "checkpoints/sam3/sam3.pt"))
# a uv venv with SAM3's few pure-Python deps (timm, ftfy, iopath, pycocotools, setuptools) next to torch
SAM3_PYTHON = os.environ.get("SAM3_PYTHON", "/leonardo_work/IscrC_EditGS/gabri/venvs/tttlrm_env/bin/python")
# Python.h for Triton's JIT when the worker's interpreter ships no headers (ABI-compatible spack python 3.11)
SAM3_PY_INCLUDE = os.environ.get(
    "SAM3_PY_INCLUDE", "/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0"
                       "/python-3.11.7-ziwh63aulhhzxksf42k5u3gnim2rbpmp/include/python3.11")


class Sam3Worker:
    def __init__(self, python=SAM3_PYTHON, repo=SAM3_REPO, checkpoint=SAM3_CHECKPOINT, runner_args=()):
        env = dict(os.environ, SAM3_PY_INCLUDE=SAM3_PY_INCLUDE, PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)  # the GW env's paths must not leak into the worker
        self.t_start, self.ready = time.time(), None
        self.proc = subprocess.Popen([python, RUNNER, "--repo", repo, "--checkpoint", checkpoint, *runner_args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"SAM3 worker exited with code {self.proc.wait()} (its log is on stderr)")
        answer = json.loads(line)
        if "error" in answer:
            raise RuntimeError(f"SAM3 worker: {answer['error']}")
        return answer

    def wait_ready(self):
        """Blocks until the model is loaded. -> runner's ready message + waited (s blocked here), since_start."""
        if self.ready is None:
            t = time.time()
            self.ready = self._read()
            self.ready.update(waited=time.time() - t, since_start=time.time() - self.t_start)
        return self.ready

    def _call(self, **request):
        self.wait_ready()
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        return self._read()

    def detect(self, image, prompt, out):
        """Text prompt on one image; out = (H, W) bool mask of the best instance. -> {found, score, scores, seconds}"""
        return self._call(op="detect", image=image, prompt=prompt, out=out)

    def track(self, frames, mask, out):
        """Track the frame-0 mask through frames/<i>.jpg; out/<i>.npy (h, w) bool. -> {frames, nonempty, seconds}"""
        return self._call(op="track", frames=frames, mask=mask, out=out)

    def close(self):
        """Polite quit once loaded; before ready (the GW side failed early) or on a broken pipe: kill, so the caller
        neither waits for the model load nor sees a worker error in place of its own."""
        if self.proc.poll() is not None:
            return
        if self.ready is None:
            return self.kill()
        try:
            self._call(op="quit")
        except (OSError, RuntimeError, ValueError):
            self.kill()
        finally:
            if self.proc.stdin and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
            self.proc.wait()

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            self.kill()


def add_worker_args(parser):
    parser.add_argument("--sam3_python", default=SAM3_PYTHON, type=str, help="Python of the SAM3 env (env SAM3_PYTHON)")
    parser.add_argument("--sam3_repo", default=SAM3_REPO, type=str, help="SAM3 repository (env SAM3_REPO)")
    parser.add_argument("--sam3_checkpoint", default=SAM3_CHECKPOINT, type=str, help="sam3.pt (env SAM3_CHECKPOINT)")
    parser.add_argument("--sam3_args", default="", type=str,
                        help="extra sam3_runner.py flags as one '=' value, comma or space separated: "
                             "--sam3_args=--no_fast_init,--no_prefetch (the comma form survives unquoted shell "
                             "expansion, e.g. slurm SEED2GS_ARGS)")


def split_runner_args(value):
    """--sam3_args value -> runner argv: quotes stripped (shell-style), then commas and spaces separate (the runner's
    flags take no values with either)."""
    return [t for tok in shlex.split(value) for t in re.split(r"[\s,]+", tok) if t]


def worker_from_argv(argv=None):
    """Sam3Worker from the --sam3_* flags of a command line (other flags ignored)."""
    parser = argparse.ArgumentParser(add_help=False)
    add_worker_args(parser)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return Sam3Worker(args.sam3_python, args.sam3_repo, args.sam3_checkpoint, split_runner_args(args.sam3_args))

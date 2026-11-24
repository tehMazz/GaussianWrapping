"""HollowSchedule: hollowing losses and prunes for GW train.py and the BEA-GS finetune.

    hollow = HollowSchedule(args, scene, gaussians, pipe)
    loss = loss + hollow.loss(iteration)
    ...                                           # inside the no-grad block, before optimizer.step
    gaussians_have_changed |= hollow.after_step(iteration)
    visible = radii > 0                           # SparseGaussianAdam: also update prior-only Gaussians
    if hollow.touched is not None: visible |= hollow.touched

From hollow_from on: lambda_fs * FreeSpacePrior every iteration (the free grid is rebuilt every
free_grid_every), lambda_entropy * opacity entropy from entropy_from, opacity decay every
opacity_decay_every when opacity_decay > 0, unseen pruning every prune_unseen_every,
see-through carving every carve_every, and finalize() at the last iteration (args.iterations).
GW train.py saves in training_report, before the no-grad hooks of the same iteration, so the
final prune reaches the saved PLY only if after_step(last) runs before that save. finalize()
can also be called on its own, e.g. after the BEA-GS finetune or AF3D distillation.
"""
import torch

from hollow.carve import carve_see_through
from hollow.free_space import FreeSpacePrior, apply_opacity_decay, opacity_entropy_loss
from hollow.ledger import compute_seen, prune


class HollowSchedule:
    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group("Hollowing")
        g.add_argument("--hollow", action="store_true", help="enable the hollowing stage")
        g.add_argument("--hollow_from", type=int, default=20_000)
        g.add_argument("--lambda_fs", type=float, default=1e-3, help="free-space prior weight")
        g.add_argument("--lambda_free", type=float, default=1.0, help="ray-carved free grid term, relative to lambda_fs")
        g.add_argument("--free_grid_every", type=int, default=5000, help="0 = build the free grid once")
        g.add_argument("--lambda_entropy", type=float, default=1e-3)
        g.add_argument("--entropy_from", type=int, default=25_000)
        g.add_argument("--opacity_decay", type=float, default=0.0, help="opacity factor, e.g. 0.995; 0 = off")
        g.add_argument("--opacity_decay_every", type=int, default=100)
        g.add_argument("--prune_unseen_every", type=int, default=2000)
        g.add_argument("--seen_tau", type=float, default=1.0, help="prune seen < tau pixels (hollow/ledger.py)")
        g.add_argument("--carve_every", type=int, default=5000)
        g.add_argument("--carve_threshold", type=float, default=0.9)
        g.add_argument("--final_prune_opacity", type=float, default=0.05)
        g.add_argument("--no_hollow_finalize", action="store_true",
                       help="skip the final prune at the last iteration (no recovery iterations follow it)")
        return g

    def __init__(self, args, scene, gaussians, pipe):
        self.args = args
        self.gaussians = gaussians
        self.pipe = pipe
        self.cameras = scene.getTrainCameras().copy()
        self.enabled = bool(getattr(args, "hollow", False))
        self.last_iteration = getattr(args, "iterations", None)
        self.prior = None
        if self.enabled and not hasattr(gaussians, "_culling"):
            gaussians.init_culling(len(self.cameras))  # prune_points needs it on a loaded model

    @property
    def touched(self):
        """(P,) bool: Gaussians the last prior loss sampled, or None."""
        if self.prior is None or self.prior.touched is None:
            return None
        if self.prior.touched.shape[0] != self.gaussians.get_xyz.shape[0]:
            return None
        return self.prior.touched

    def _active(self, iteration):
        return self.enabled and iteration >= self.args.hollow_from

    def _due(self, iteration, every):
        return every > 0 and iteration > self.args.hollow_from and (iteration - self.args.hollow_from) % every == 0

    def loss(self, iteration):
        if not self._active(iteration):
            return 0.0
        a = self.args
        loss = 0.0
        if a.lambda_fs > 0:
            if self.prior is None:
                self.prior = FreeSpacePrior(self.cameras, self.gaussians, lambda_free=a.lambda_free)
                if a.lambda_free > 0:
                    self._update_grid()
            loss = a.lambda_fs * self.prior.loss(self.gaussians)
        if a.lambda_entropy > 0 and iteration >= a.entropy_from:
            loss = loss + a.lambda_entropy * opacity_entropy_loss(self.gaussians)
        return loss

    @torch.no_grad()
    def after_step(self, iteration):
        """Scheduled decay / prunes / grid rebuild; True when Gaussians were removed."""
        if not self._active(iteration):
            return False
        a = self.args
        if a.opacity_decay > 0 and iteration % a.opacity_decay_every == 0:
            apply_opacity_decay(self.gaussians, a.opacity_decay)
        if iteration == self.last_iteration:
            return False if getattr(a, "no_hollow_finalize", False) else self.finalize() > 0

        n, seen = 0, None
        if self._due(iteration, a.prune_unseen_every):
            seen = compute_seen(self.cameras, self.gaussians, self.pipe)
            unseen = seen < a.seen_tau
            n_unseen = prune(self.gaussians, unseen)
            seen = seen[~unseen]
            n += n_unseen
            print(f"[INFO] Hollow: pruned {n_unseen} unseen Gaussians (seen < {a.seen_tau}) at iteration {iteration}.")
        if self._due(iteration, a.carve_every):
            n_carved = prune(self.gaussians, carve_see_through(
                self.gaussians, self.cameras, self.pipe, a.carve_threshold, seen=seen))
            n += n_carved
            print(f"[INFO] Hollow: carved {n_carved} see-through Gaussians at iteration {iteration}.")
        if n:
            self._refresh()
        if self.prior is not None and a.lambda_free > 0 and self._due(iteration, a.free_grid_every):
            self._update_grid()
        return n > 0

    @torch.no_grad()
    def finalize(self):
        """One pass of all prunes: unseen, see-through, filtered opacity < final_prune_opacity."""
        a = self.args
        seen = compute_seen(self.cameras, self.gaussians, self.pipe)
        drop = seen < a.seen_tau
        drop |= carve_see_through(self.gaussians, self.cameras, self.pipe, a.carve_threshold, seen=seen)
        drop |= self.gaussians.get_opacity_with_3D_filter[:, 0] < a.final_prune_opacity
        n = prune(self.gaussians, drop)
        print(f"[INFO] Hollow: final prune removed {n} Gaussians, {self.gaussians.get_xyz.shape[0]} left.")
        if n:
            self._refresh()
        return n

    def _refresh(self):
        if self.gaussians.use_mip_filter:
            self.gaussians.compute_3D_filter(cameras=self.cameras)
        if self.prior is not None:
            self.prior.touched = None

    def _update_grid(self):
        grid = self.prior.update_grid(self.gaussians, self.pipe)
        print(f"[INFO] Hollow: free grid has {grid.keys.numel()} voxels of size {grid.size:.4g}.")

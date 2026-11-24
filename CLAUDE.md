# BEA-GS on Gaussian Wrapping

Read `HANDOFF.md` first (goal, decisions, results, open problems, next steps), then `GWBEAGS.md` (setup, scripts, flags).

- Python code lives in `gaussian_wrapping/`; run scripts from the repo root as `python gaussian_wrapping/<script>.py`.
- Leonardo: the login node has no GPU. GPU work goes through `slurm/*.sh` (account `IscrC_ESGS`); smoke tests on the
  debug QoS (`slurm/gw_smoke.sh`). Env and modules: `slurm/common.sh`.
- GPU budget: one scene per dataset, smoke first, ask before launching full runs.
- Outputs are flat: `output/<method>/<scene>/<mode>`, never nested inside a model dir.
- Show results as renders (`beags_render_labels.py`: >= 5 views, label 0 and label 1 alone), not only PLYs.
- Images are always loaded on the CPU (`--data_device cpu`, now the `ModelParams` default); never `cuda`.
- BakerH VM run: `HANDOFF_BAKERH.md`. BakerH VM scripts follow ArtiFixer's `scripts/bakerh/run_all.sh` REPORT.txt pattern.
- Commit or push only when asked.

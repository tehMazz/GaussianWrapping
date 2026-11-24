# Handoff: BakerH run on the VM (BEA-GS on Gaussian Wrapping)

Written 2026-11-02 on Leonardo for whoever (person or Claude session) runs this repo on the BakerH VM. Read this file
first; `HANDOFF.md` (project state) and `GWBEAGS.md` (scripts, flags) hold the background.

## 1. The task
On each of the 7 BakerH scenes, one after another on one GPU:
1. train Gaussian Wrapping 30k **with the bilateral grid** and hollowing;
2. apply BEA-GS (labels from the SAM3 masks, 3000-it finetune, dilation weight 0.1);
3. render, for >= 5 views (8 by default), the **extraction** (label 1 alone) and the **background** (label 0 alone, i.e.
   the scene after removal), next to the photo and the full render.

Scenes: `TA_BLUE_MOTOR PCV TA_TURBINE Compressor G35 COGEN1 COGEN2`.
The mesh and ArtiFixer are **not** part of this run (opt-in with `FULL=1`, section 6).

## 2. Rules
- **Images are always loaded on the CPU** (`--data_device cpu`). It is the default in `gaussian_wrapping/arguments/__init__.py`
  and in `GW_TRAIN_ARGS`; every later script inherits it from the model's `cfg_args`. Never pass `--data_device cuda`:
  Compressor has 923 photos at 1600 px, which do not fit on the GPU next to the model.
- **One GPU** (`GPU=1` by default). Do not start a second run while one is going (`run_all.sh` refuses).
- **Outputs are flat:** `output/<method>/<scene>/<mode>`, never nested inside a model dir.
- **Results are renders, not only PLYs:** send back the `showcase/` grids (section 5).
- **Do not change the recipe** without asking the user (Alessio). If a phase fails, send `REPORT.txt` back instead of
  patching around it.
- Commit or push only when asked.

## 3. Get the code onto the VM
Nothing of the BEA-GS pipeline is committed yet (`HANDOFF.md` section 9), so a `git clone` has none of it. Rsync the
whole working tree from Leonardo, submodules included:
```bash
rsync -a --exclude output/ --exclude logs/ --exclude '__pycache__/' --exclude .venv/ \
    <leonardo>:/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/BEA-GS_GWrap/ \
    /workspace/amazzucchelli/fbk-3dworld/BEA-GS_GWrap/
```
`setup.sh` step 0 checks that the tree is complete and stops if it is not.

What the scripts expect on the VM (all overridable in `scripts/bakerh/common.sh`):
| what | default |
|---|---|
| data | `BakerH_undis/<scene>/{images_1600,sparse}` under `datasets/`, else `../BEA-GS_FSplat/datasets/`, else `$FBK/BEA-GS_FSplat/datasets/` (`DATA`) |
| SAM3 masks | `$FBK/FlashSplat/sam3_single_object_anchor_tracking/<scene>/{full,0_400,400_800}/masks_npy` (`SAM_MASKS`) |
| `FBK` | `/workspace/amazzucchelli/fbk-3dworld` |
| venv | `<repo>/.venv` (built by `setup.sh`: Python 3.9, torch 2.3.1+cu121, the CUDA extensions, LPIPS weights) |

## 4. Run it
```bash
cd <repo on the VM>
bash scripts/bakerh/run_all.sh          # setup.sh, then the 7 scenes; detaches and survives logout
tail -f output/bakerh_logs/latest/all.log
```
- Stop: `kill -- -$(cat output/bakerh_logs/latest/pgid)`.
- Rerun: the same command. Finished phases are skipped in seconds; a failed scene is skipped and the next one runs.
- Subset: `SCENES="PCV G35" bash scripts/bakerh/run_all.sh`. Skip setup once it passed: `SKIP_SETUP=1`.
- Redo a phase: `FORCE=1 PHASES=renders,showcase bash scripts/bakerh/run_scene.sh PCV`.

Phases per scene (default, `run_scene.sh` header has the details):
`check, masks, train, export, prepare, finetune, labels, export_beags, renders, showcase, metrics, hollow`.

Time (estimate, never measured at 1600 px): kitchen at 779x520 took 1 h end to end on Leonardo, so expect 1-2 h per
scene, more for Compressor (923 photos), and 10-15 h for all 7. `REPORT.txt` records the real per-phase times.

Training args (`GW_TRAIN_ARGS`): `--rasterizer ours --feature_dc_lr 0.0013 --feature_rest_lr 0.00011 --data_device cpu
--N_max_gaussians 6000000 --no-exposure_compensation --appearance bilagrid --hollow`.

## 4b. Step by step: `scripts/bakerh/stages/` (added 2026-11-24, never run on the VM)
One script per stage, default `SCENES="TA_BLUE_MOTOR Compressor"`. Run a stage, check what its `CHECK.txt` lists,
then run the next one. Each stage is `run_all.sh` with a fixed `PHASES` list: detached, finished phases are skipped,
logs in `output/bakerh_logs/<stamp>_stage<N>/{REPORT.txt,CHECK.txt}`.

| script | phases | what to look at |
|---|---|---|
| `stage0_setup.sh` | setup.sh + check | `SETUP OK` |
| `stage1_train.sh` | check, masks, train, export, metrics | 30k test PSNR |
| `stage2_segment.sh` | prepare, finetune, labels | IoU vs SAM3 at 30k / 33k |
| `stage2b_seed2gs.sh` (opt.) | seed2gs (needs `SAM3_PYTHON`) | Seed2GS mask IoU vs SAM3 |
| `stage3_extract.sh` | export_beags, renders, showcase, metrics, hollow | `showcase/<scene>_grid_all.jpg` |
| `stage4_mesh.sh` | mesh | `mesh/hole_1.ply` |
| `stage4b_meshocc.sh` (opt.) | mesh_train, finetune_meshocc, labels_meshocc, renders_meshocc, metrics | `<scene>_meshocc_grid_all.jpg`, IoU vs stage 2 |
| `stage5_holes.sh` | afinputs (`AF_UNSEEN=on`), artifixer `STAGES=prepare,removal,split`, holes | `output/artifixer/<scene>/<dir>/hole_grid.jpg` |
| `stage6_artifixer.sh` | afinputs, artifixer (all stages) | inpainted frames, `af3d/export/point_cloud.ply` |
| `stage7_variant.sh` | stages 1-3 with `VARIANT=conf` or `gridfix` | own `METHOD`, compare with the baseline |

- `VARIANT=conf|gridfix` works on every stage: `METHOD=gw_<variant>_beags_dw0.1`, ArtiFixer dir `<AF_VARIANT>_<variant>`.
  `conf` = `train.py --confidence` (the stage-5 confidence hole needs it); `gridfix` = `GRIDFIX_ARGS` (identity + mean
  pulls 10, grid off after 25000, `--lambda_raw 0.2`: my choice of values, untested at 30k).
- Virtual views: `AF_VARIANT=vanilla_object|vanilla_vidsplat|vanilla_travel bash .../stage6_artifixer.sh` (novel camera
  path, no removal). The default `normal` removal works on the photo frames only.
- Stage 5 before stage 4 gives an alpha hole; to switch to the mesh hole later: `FORCE=1` on stage 5.
- `common.sh` has a fallback caption for Compressor (ours), used only when ArtiFixer's root has no `caption.h5`.

## 4c. Novel views only: `stages/nv1_base.sh` -> `nv2_paths.sh` -> `nv3_artifixer.sh` (added 2026-11-24)
Tests the GW + ArtiFixer combination on novel camera paths (`AF_VARIANT=vanilla_<t>`, no removal):
1. `nv1_base.sh`: GW 30k (grid + hollowing), export, BEA-GS prepare + finetune (labels for the object path; the finetune
   re-fits the grid's colour shift), metrics. ArtiFixer starts from the 33k model.
2. `nv2_paths.sh`: per path in `TRAJS` (default `object vidsplat travel`): afinputs + ArtiFixer-GS
   `STAGES=trajectory,prepare` + `trajcands` -> `trajectory_input.png` (top view) and `path_candidates.jpg` (16 GW renders
   on the path). No diffusion.
3. `nv3_artifixer.sh`: per path: the remaining ArtiFixer-GS stages (infer, handoff, ArtiFixer3D, ArtiFixer3D+) +
   `trajgrid` -> `output/<method>/showcase/<scene>_vanilla_<t>_grid.jpg` (GW render | ArtiFixer | ArtiFixer3D render |
   ArtiFixer3D+).
The paths run one after another in one detached chain; each gets its own `<stamp>_stage<nv>_<t>/REPORT.txt`. Compressor
is capped at 121 path frames (`COMPRESSOR_TRAJ_FRAMES`, = TRAJ_MAX_FRAMES; 923 frames ran out of memory in removal).

## 5. What to send back
1. `output/bakerh_logs/latest/REPORT.txt`: setup result, per-phase status and time, peak GPU memory, PSNR / SSIM / LPIPS,
   BEA-GS IoU vs the SAM3 masks, Gaussians per label, hollow metrics, and a one-line-per-scene table.
2. `output/gw_beags_dw0.1/showcase/<scene>_grid_all.jpg`, one per scene. Each row is one view:
   **photo | full render | label_0 = background after removal | label_1 = extracted object** (white background).
   Per-view PNGs: `showcase/<scene>/{grid,full,label_0,label_1}/`.

Other outputs per scene, in `output/gw_beags_dw0.1/<scene>/`: `masks`, `train` (30k), `export` (30k PLY),
`beags` (33k), `export_beags/{point_cloud,label_0,label_1}.ply`, `renders_labels_33000`.

## 6. Known issues (read before judging the numbers)
- **The bilateral grid lowers raw-render PSNR** (`HANDOFF.md` 7.1). On Mip-NeRF 360 kitchen the 30k model scored 26.8
  vs 32.4 for FastGS; the loss sees the grid-corrected render, eval sees the raw one, so raw colours drift per view.
  The BEA-GS finetune (no grid) re-fits colour and recovers ~3 dB. So: judge the 33k (`beags`) renders and numbers, and
  expect the 30k export to look colour-shifted. The drift-fix flags (`--bilagrid_identity_weight`, `--bilagrid_until`,
  `--lambda_raw`) exist but are untested; do not add them unless the user asks.
- **Removal residue:** on kitchen, boundary fragments and a dark smear under the object stayed in label 0. Look for
  the same in the label_0 column.
- **BakerH has no GT masks:** the `labels` phase scores agreement with the SAM3 input masks, not accuracy.
- **Never run on the VM before.** `setup.sh` and the scene scripts were dry-checked on Leonardo against the kitchen
  outputs only. The first failures will most likely be in `setup.sh` (CUDA extension builds). `REPORT.txt` says which
  step and why.
- `FULL=1` adds the multi-label mesh + hole, the ArtiFixer inputs, and the ArtiFixer diffusion on `ARTIFIXER_SCENES`
  (`TA_BLUE_MOTOR PCV`). The mesh takes 1-3 h per scene. Only on request.

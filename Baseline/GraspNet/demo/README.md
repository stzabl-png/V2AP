# GraspNet-Demo

Local GraspNet inference → V2AP-compatible output for Razor deployment.

## What this replaces

```
V2AP Full Pipeline:          GraspNet-Demo:
T1 validate                  ─── (same input format)
T2 SAM segmentation     →    ─── (skipped: no mesh needed)
T3 SAM3D mesh           →    ─── (skipped)
T4 metric scale         →    ─── (skipped)
T5 FoundationPose       →    ─── (skipped)
T6 grasp_pose           →    graspnet_scene_infer.py  ✅
T7 status.json          →    written automatically     ✅
```

## Usage

```bash
cd /home/lyh/Project/V2AP

# ── Scene-level inference (recommended) ──────────────────────
python -m graspnet_demo.run_graspnet_demo \
    --session-dir /media/lyh/KINGSTON/20260603_165343_chips \
    --scene-level

# ── Object-level inference (with mask) ───────────────────────
python -m graspnet_demo.run_graspnet_demo \
    --session-dir /media/lyh/KINGSTON/20260603_165343_chips \
    --seg mask --mask-path .../mask.png

# ── Dry-run: inspect poses ────────────────────────────────────
python -m graspnet_demo.run_graspnet_demo \
    --session-dir /media/lyh/KINGSTON/20260603_165343_chips \
    --dry-run

# ── Validate format ───────────────────────────────────────────
python -m graspnet_demo.validate_candidates \
    --session-dir /media/lyh/KINGSTON/20260603_165343_chips
```

## Output structure

```
<session>/output/
├── status.json                  # V2AP run_auto_grasp reads this first
└── inference/
    └── candidates.json          # V2AP-compatible (schema_version: 1.1)
```

## Deploy to Razor

```bash
SESSION=20260603_165343_chips
RAZOR=razor  # or IP

# 1. Copy output to Razor
rsync -avz /media/lyh/KINGSTON/$SESSION/output/ \
    $RAZOR:~/V2AP-demo/demo/phase2/sessions/$SESSION/output/

# 2. Dry-run on Razor (verify poses)
ssh $RAZOR "cd V2AP-demo && python demo/phase2/run_auto_grasp.py \
    --session-id $SESSION --dry-run"

# 3. Execute grasp
ssh $RAZOR "cd V2AP-demo && python demo/phase2/run_auto_grasp.py \
    --session-id $SESSION --debug"
```

## Key design decisions

- **T_base_mesh = eye(4)**: Our `grasp_point` is already in the robot base frame.
  V2AP's `retarget.py` computes `T_base_pinch = T_base_mesh @ T_mesh_pinch`.
  With `T_base_mesh=I`, this reduces to `T_base_pinch = T_mesh_pinch`, which
  means the `grasp_point` *is* the pinch position in base frame. ✅

- **rotation col 2 = approach**: Both GraspNet and UCB use the same convention.
  Pre-grasp = retreat 0.15m along `-approach`. ✅

- **No ee_retarget calibration needed for dry-run**. For live execution on Razor,
  run `calibrate_ee_retarget.py` once to measure `T_ee_pinch`.

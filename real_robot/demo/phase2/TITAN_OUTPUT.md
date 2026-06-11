# Titan `output/` package (for Razor)

**Titan** writes this tree after Phase 2 perception (T2–T6). **Razor** rsyncs `output/` and runs retarget + IK.

**Automation (Titan team):** [SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md) — SSH/rsync, `python -m demo.pipeline.process_razor_session`, `status.json`.

This doc describes **what Titan produces**, coordinate frames, and how Razor consumes `candidates.json` (not files that Razor generates).

Full input/output spec: [README.md](README.md).

---

## 1. What to rsync

From Titan session root `…/<session_id>/`:

```bash
# Example (adjust hosts/paths)
rsync -avz vision@<titan>:.../demo/sessions/<session_id>/output/ \
  ./demo/phase2/sessions/<session_id>/output/
```

Minimum for automatic grasp (**required**):

| Path | Purpose |
|------|---------|
| `output/status.json` | **Read first.** `success: true` means safe to run grasp |
| `output/inference/candidates.json` | Ranked grasp hypotheses + registration |
| `output/register/T_base_mesh.json` | mesh → robot **base** (must match candidates) |
| `output/mesh/object_base_aligned.glb` | Mesh in same frame as candidates (`base_aligned`) |

Strongly recommended (debug / re-run):

| Path | Purpose |
|------|---------|
| `output/register/T_cam_mesh.json` | mesh → camera (debug, consistency check) |
| `output/segment/mask.png` | Segmentation used by FP / scale |
| `output/mesh/scale.json` | Metric scale metadata |
| `output/register/foundationpose_meta.json` | FP timing, warnings |
| `output/inference/affordance_grasp.hdf5` | Optional; `candidates.json` is enough for Razor |
| `output/vis/T5_foundationpose_overlay.png`, `T6_grasp_vis.png` | Human review |

You do **not** need Titan’s `input/` on Razor if you already have the same session from capture — but keeping the full session folder is fine.

---

## 2. `status.json` (T7)

Written by:

```bash
python demo/scripts/T7/write_status.py --session-dir demo/sessions/<session_id>
```

**Before running grasp on Razor:**

```python
status = json.load(open("output/status.json"))
assert status["success"] is True
```

| Field | Meaning |
|-------|---------|
| `success` | `true` = all pipeline steps present and consistency checks passed |
| `steps` | Per-step: `segment`, `sam3d`, `scale`, `foundationpose`, `grasp_pose` → `"ok"` / `"fail"` |
| `warnings` | Non-fatal (e.g. table height vs mesh Z, scale clamp, distortion) |
| `errors` | Fatal — do not grasp |
| `package.required_for_grasp` | Minimal file list |
| `titan.n_candidates` | Number of exported candidates |

---

## 3. Coordinate frames (important)

Candidates are **not** stored directly in `R_ee` or a ready-to-IK robot TCP pose.

| Frame | Where defined | Used for |
|-------|----------------|----------|
| **mesh** (`base_aligned`) | `object_base_aligned.glb` local coords | `grasp_point`, `rotation` in each candidate |
| **camera** | `register/T_cam_mesh.json` | Debug, reprojection |
| **base** | `input/calib/extrinsics.json` → `T_base_cam` | Robot arm IK target chain |

**Composition (must hold):**

```text
T_base_mesh = T_base_cam @ T_cam_mesh     # also in candidates.json
p_base      = T_base_mesh @ [p_mesh, 1]   # homogeneous
```

Each candidate:

```json
{
  "grasp_point": [x, y, z],
  "rotation": [[...], [...], [...]],
  "position_panda_hand": [...]
}
```

| Field | Frame | Razor usage |
|-------|-------|-------------|
| `grasp_point` | **`base_aligned`** | Virtual pinch center; same frame as `object_base_aligned.glb` |
| `rotation` | **`base_aligned`** | Columns: `finger_open`, `y_body`, `approach` (col index 2 = approach axis) |
| `position_panda_hand` | **`base_aligned`** | Franka `panda_hand` TCP — **do not use for Dexmate IK** |
| `T_base_mesh` | **base ← mesh** | Multiply to get pinch pose in base |

---

## 4. Razor retarget (not on Titan)

Titan does **not** know Sharpa hand kinematics or your `R_ee` calibration. Retarget on Razor (Phase 2 `run_auto_grasp.py` pattern):

### Step R1 — Pinch in base

```python
import numpy as np

T_base_mesh = np.array(candidates["T_base_mesh"])  # 4×4
R_mesh = np.array(cand["rotation"])
t_mesh = np.array(cand["grasp_point"])

T_mesh_pinch = np.eye(4)
T_mesh_pinch[:3, :3] = R_mesh
T_mesh_pinch[:3, 3] = t_mesh

T_base_pinch = T_base_mesh @ T_mesh_pinch
```

### Step R2 — Pinch → `R_ee` (open-grip IK on Razor)

Default path: **open-hand** thumb/index FK constraints + Gauss–Newton IK (`open_grip_retarget_geometry.py`, `pinch_ik.py`) — not closed-hand `T_ee_pinch` bridge. See `run_auto_grasp.py` and `calib/ee_retarget.yaml` (`open_pinch_forward_offset_m`, etc.).

### Step R3 — Motion

- **Pre-grasp:** retreat along **−approach** in base by `conventions.pre_grasp_offset_m` (default **0.15 m**), matching UCB sim / Phase 1.
- **IK:** target `T_base_ee`, not `position_panda_hand`.
- **Hand:** open profile → approach → stall close → lift (Phase 1 `hand_tuner` / `run_grasp.py`).
- **Ranking:** try candidates by `rank` order; **re-rank by IK feasibility** on Dexmate (UCB score is Franka-biased).

---

## 5. File reference

### `output/inference/candidates.json`

Top-level fields:

| Field | Description |
|-------|-------------|
| `schema_version` | `"1.1"` |
| `mesh_frame` | `"base_aligned"` — use with `object_base_aligned.glb` |
| `registration` | `T_cam_mesh`, `T_base_mesh`, method `foundationpose` |
| `conventions` | `rotation_columns`, `ucb_tcp_offset_m` (0.105), `pre_grasp_offset_m` (0.15) |
| `mesh_aabb_min_m` / `mesh_aabb_max_m` | Optional; Razor `object_obstacle.py` (can recompute from GLB) |
| `candidates[]` | Sorted by `rank` (0 = best score) |

### `output/register/T_base_mesh.json`

Authoritative **base ← mesh** for execution. Must match `candidates.json` `T_base_mesh` (T7 checks this).

### `output/mesh/object_base_aligned.glb`

Same physical pose as FP; mesh local axes aligned to robot base (rotation ≈ identity in `T_base_mesh`).

### `output/mesh/object_scaled.glb`

Metric scaled mesh in **FP/SAM3D frame** (before base-axis alignment). Use **aligned** GLB for grasp, not this file, unless you recompose transforms yourself.

---

## 6. Typical Razor workflow

1. Capture session on Razor → `input/` (already on laptop).
2. Rsync `input/` → Titan; run T2–T6 on Titan.
3. Rsync `output/` ← Titan.
4. Run T7 on Titan **or** trust Titan already ran `write_status.py`.
5. On Razor: read `status.json` → if `success`, run Phase 2 auto grasp with `candidates.json` + `ee_retarget.yaml`.
6. If all IK fail: inspect `T6_grasp_vis.png`, adjust mask/mesh on Titan, or tune retarget.

---

## 7. What Titan does *not* send

| Item | Where it lives |
|------|----------------|
| `T_ee_pinch` / Sharpa → `R_ee` | Razor `demo/phase2/calib/ee_retarget.yaml` |
| IK, collision, trajectory | V2AP-demo `run_auto_grasp.py` |
| Phase 1 start/grasp YAML | `demo/phase1/objects/` |

---

## 8. Questions / changelog

- Titan scripts index: [demo/scripts/README.md](scripts/README.md)
- T7 finalize: [demo/scripts/T7/README.md](scripts/T7/README.md)

If `status.json` reports warnings but `success: true`, grasp may still work; treat table-height / distortion warnings as review items, not automatic abort.

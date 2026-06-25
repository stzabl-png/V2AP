# Baseline1 v3 — CAD-First G-Frame Plan

> **Goal**: Get a closed-loop DP3 policy that can pick up an object in IsaacSim
> (dz > 3cm), starting from human videos via supervised trajectory imitation.
>
> **Strategy**: Maximum derisk by simplification:
> 1. **CAD mesh** (not SAM3D) → eliminates 4 of 6 bugs
> 2. **DexYCB only** (not DexYCB+OakInk) → cleaner GT, 6 cameras for cross-cam test, AprilTag for table_z
> 3. **Single object first** (ycb_dex_01, master_chef_can) → fast iteration, clear signal
> 4. Expand only after each phase succeeds: DexYCB-1obj → DexYCB-all → +OakInk → SAM3D

---

## 0. Why this plan

### Current state (v2, May 14-16)
- Trained 100 epoch DP3 on combined DexYCB+OakInk (9652 ep, R_align + GT pose)
- ckpt: `Baseline1/dp3_runs/all_combined_v2/checkpoints/epoch=0012-test_mean_score=-0.005.ckpt`
- IsaacSim eval: 4 rollouts on A16013, **all failed (dz ≈ 0 / NaN)**
- Root cause analysis identified 6 bugs (see §1.1)

### Decisions confirmed
1. **Use CAD mesh** (DexYCB textured.obj) instead of SAM3D — eliminates 4 of 6 bugs
2. **Use GT pose** (DexYCB pose_y) instead of FoundationPose — natural fit with CAD
3. **G-frame Object-Centric** (gravity-aligned, table-anchored) — solves cross-camera and train/sim frame mismatch
4. **yaw augmentation** during training — handles +X direction ambiguity (defer to Phase 2 if Phase 1 success without it)
5. **DexYCB only for now** — defer OakInk until DexYCB pipeline validated
6. **Single-object pilot first** (ycb_dex_01) — derisk before scaling
7. **No real-robot deployment** in scope — sim-only

### Bugs eliminated by this plan
- ✅ Bug A: R_align rotation mismatch → CAD has no R_align
- ✅ Bug D: OakInk scale variation → CAD in real units
- ✅ Bug E: ycb_dex_20 SAM3D quality → CAD always clean
- ✅ Bug F: R_align rotation symmetry ambiguity → no R_align

### Bugs solved by G-frame + yaw augmentation
- ✅ Bug B: train (C-frame OpenCV) vs sim (W-frame IsaacSim) axes mismatch
- ✅ Bug C: spawn pose distribution — partial solution via yaw augmentation

---

## 1. Frame Design

### 1.1 Coordinate frames (all in metres)

| Frame | Origin | Axes | Used by |
|---|---|---|---|
| **M-frame** | CAD mesh's own centroid (designer-defined) | CAD canonical | mesh files |
| **C-frame** | Optical centre of camera k | OpenCV (+X right, +Y down, +Z forward) | dataset GT pose, MANO joints |
| **W-frame** | Master camera optical centre (DexYCB) / dataset-defined (OakInk) | Inherits master C-frame axes (+Y gravity in DexYCB) | extrinsics file |
| **G-frame** | `[obj_center_t0.x_W, obj_center_t0.y_W, table_z_W]` | +Z up (gravity-aligned), +X = arbitrary (yaw-augmented at training) | **policy I/O** |

### 1.2 G-frame origin computation

**G-frame axes constants (verified May 17)**:

| Axis | Direction (DexYCB) | Physical meaning |
|---|---|---|
| +X_G | +X_W (master cam right) | lab horizontal (some direction parallel to table) |
| +Y_G | -Z_W (master cam back) | lab horizontal (other direction parallel to table) |
| +Z_G | -Y_W (master cam up) | reverse gravity (up) |


**Per-episode origin in W-frame** (before applying W→G rotation):
- `(x_origin_W, y_origin_W)` = object centroid XY at t=0, projected onto table
- `z_origin_W` = table surface position in the gravity axis of W-frame

**Table height estimation — object lowest-point method** (D4 decision: simpler self-contained, no AprilTag dependency):

```python
def estimate_table_y_W(sessions, cad_mesh_per_obj, n_frames=5):
    """Multi-object, multi-frame lowest-point estimate (D4).
    Object at rest on table at t=0 → its 'lowest in gravity direction' ≈ table.
    DexYCB W +Y down: 'lowest' = largest +Y → use 99th percentile of pc_W.y
    """
    lowest_per_episode = []
    for session in sessions:
        for ep in session.episodes:
            cad_mesh = cad_mesh_per_obj[ep.obj_id]
            per_frame = []
            for t in range(min(n_frames, len(ep))):
                pc_W = pose_y_W_list[t] @ cad_mesh
                per_frame.append(np.percentile(pc_W[:, 1], 99))
            lowest_per_episode.append(np.median(per_frame))
    return float(np.median(lowest_per_episode))   # median across all episodes in same camera setup
```

**Aggregation strategy**:
- Within episode: median of first 5 frames (object stationary)
- Across episodes (same camera setup): median (robust to outliers — episodes where object isn't on table)
- One table_y_W value per "extrinsics batch" (DexYCB sessions are grouped by extrinsics calibration ID in meta.yml)

**Sanity self-check** (without AprilTag): table_y_W should be:
- Consistent across same extrinsics batch (std < 1cm)
- Roughly 0.6-1.0m if master cam is ~1m above table (rough sanity, not rigorous)
- Negative would indicate gravity sign wrong (master cam +Y down → table is BELOW master in +Y direction → positive value expected)

**Sim**: `table_z_sim = TABLE_TOP_Z = 0.80` (constant, defined by sim setup)

### 1.3 Yaw handling

G-frame's +Z = gravity is physical. +X direction is **arbitrary** (any rotation around +Z preserves gravity).

**Phase 1**: yaw augmentation **disabled** (D3). Use a fixed +X convention (e.g., master camera's forward direction projected to horizontal plane).

**Phase 2+** (only if needed): bounded yaw augmentation, NOT full 360°.

Full 360° is wrong: Franka has fixed base, can't easily reach from arbitrary directions. A 180° rotation = "approach from behind Franka", physically infeasible. Augmentation should respect the workspace constraint:

```python
# Phase 2+ yaw range options (pick based on Franka workspace):
yaw_range = (-np.pi/6, np.pi/6)        # ±30° (conservative)
# OR (-np.pi/3, np.pi/3)               # ±60° (moderate)
# OR define +X = robot-facing direction (deterministic, no augmentation)
```

If +X is defined as robot-facing direction (i.e., from robot base toward task area), **no yaw augmentation needed** — sim's Franka base direction matches training convention automatically.

**Sim eval**: NO yaw applied. Sim's natural +X (set to match training convention) is used as-is.

---

## 2. Pipeline Changes

### 2.1 `Baseline1/retarget_human_to_ee.py` (major rewrite)

**Remove**:
- `compute_sam3d_align.py` dependency (no R_align lookup)
- `scale.json` reading (CAD has no scale_factor)
- All `_ALIGN_WARNED` logic

**Add**:
- DexYCB extrinsics loader: `load_extrinsics(extrinsics_id) → dict[cam_serial, T_W←C]`
- OakInk extrinsics loader: per-frame `cam_extr` from `image/anno/general_info/*.pkl`
- CAD mesh path resolver:
  - DexYCB: `RawData/.../dexycb/models/{ycb_name}/textured.obj`
  - OakInk: `RawData/.../oakink_v1/image/obj/{obj_id}.obj`
- Frame transform: `C-frame → W-frame → G-frame`
- Constant rotation `R_W→G`: **NOT hardcoded** — determined empirically via Gate 1b
  - The text "90° around +X" was a guess; sign and axis must be confirmed by checking that the rotation maps W-frame gravity vector to G-frame `(0, 0, -1)`. Could be `R_x(±90)` or composition; let the gate decide.
  - DexYCB: verify with apriltag (known to be below master cam, gravity direction known)
  - OakInk: verify separately (different lab, may differ)
- Table_z estimator (per episode, see §1.2)
- G-frame origin = `[obj_xy, table_z]`

**Full SE(3) transform for ALL data (critical detail — easy to miss)**:

Every position AND orientation must be transformed. Doing only position translation is wrong. Position needs origin subtraction AFTER frame rotation; rotation/quaternion needs frame composition only (no translation).

**Extrinsics convention — VERIFIED by Gate 1a (May 17)**:

DexYCB's `extrinsics.yml` stores **T_W←C** (camera's pose in world frame), i.e., to convert a C-frame point to W-frame: `p_W = R_ext @ p_C + t_ext`. The matrix is applied **directly** (no inversion or transpose).

Verified empirically: same physical object viewed from 2 cameras → byte-equivalent W-frame pose (Δ pos = 0.00mm, Δ rot = 0.01°) only under this convention. Reverse convention (T_C←W) gives 200mm mismatch.

Master cam (`840412060917`) is identity → master cam IS the W-frame origin.

```python
# Load extrinsics for camera c (3x4 matrix, packed as M = [R | t]):
M = np.array(extrinsics_yml[c]).reshape(3, 4)
R_WC = M[:, :3]   # rotation: maps C → W (i.e., R in p_W = R @ p_C + t)
t_WC = M[:, 3]    # translation: camera origin in W-frame

# Apply directly: convert a C-frame point to W-frame
p_W = R_WC @ p_C + t_WC

# To convert object pose (T_C←obj is per-frame, 4x4) to T_W←obj:
T_W_obj = T_W_C @ T_C_obj   # matrix multiply directly, no inversion
```

**OakInk**: `cam_extr` is 4×4 stored per-frame; convention to be re-verified by separate Gate 1a (likely also `T_W←C` but check).

**Compose frames (W → G is a constant per dataset)**:

```python
# DexYCB: R_W→G is determined by Gate 1b (NOT hardcoded — see below)
R_WG = R_W_to_G_for_DexYCB    # 3x3, verified via Gate 1b
origin_W = compute_episode_origin_W(...)  # [obj_centroid_xy_W, table_z_W]

# Position-like data (subtract origin in W-frame, THEN rotate):
joint_3d_G   = R_WG @ (R_WC @ joint_3d_C + t_WC - origin_W)
pose_y_pos_G = R_WG @ (R_WC @ pose_y_pos_C + t_WC - origin_W)
ee_pos_G     = R_WG @ (R_WC @ ee_pos_C + t_WC - origin_W)

# Orientation-like (rotation only, no translation):
R_compose = R_WG @ R_WC                                  # composite frame change for orientations
ee_R_G   = R_compose @ ee_R_C
ee_quat_G = R_to_quat(R_compose @ quat_to_R(ee_quat_C))
pose_y_R_G = R_compose @ pose_y_R_C

# wrist_pose 4×4 metadata:
wrist_R_G = R_compose @ wrist_R_C
wrist_t_G = R_WG @ (R_WC @ wrist_t_C + t_WC - origin_W)
```

**Why "subtract origin BEFORE rotation" or "AFTER"?** Both work if you flip the formulas correctly. The form above subtracts in W-frame, then rotates to G-frame. Equivalent alternative: rotate first, then subtract `R_WG @ origin_W`. Pick one convention and stick to it.

**Verification gates** (each must pass):
- `||v_G|| == ||v_C||` for any free vector (translation-invariant length, rigid transform preserves length)
- Master DexYCB camera position in W-frame `== (0, 0, 0)` (master is identity)
- Apply `R_WG` to a known W-frame gravity vector → should give G-frame gravity direction (see Gate 1b)

**Keep**:
- MANO joints → EE retarget logic (operates AFTER frame transform, on G-frame joints)
- Gripper 0→1 step logic (proximity is rigid-invariant)
- Quat sign continuity (apply AFTER frame transform, along the new G-frame trajectory)

### 2.2 New CAD→USD converter (small new script)

`Baseline1/tools/convert_cad_usd.py`:
- Input: CAD `.obj` file
- Output: `output/obj_usd_cad/{ds}/{obj_id}.usd`
- No scale, no rotation (CAD already in canonical + real metres)
- Apply `Z-up` USD convention
- ~50 lines, partner's `tools/convert_obj_usd.py` is the template

### 2.3 `sim/eval_dp3_policy.py` (adapt to G-frame)

**Change**:
- `find_usd`: point to `obj_usd_cad/` instead of `obj_usd/`
- `load_cad_pts` (replaces `load_sam3d_pts`): load CAD `.obj`, sample 4096 points, no scale, no canonical_rotation
- `get_object_scale`: always returns 1.0 for CAD, can be removed
- `setup_scene`: `scale=[1.0, 1.0, 1.0]` always
- `build_obs`: G-frame origin = `[obj_center_xy_W, TABLE_TOP_Z]`

**Full G-frame I/O (matching training)**:

IsaacSim W-frame is already +Z up (= G-frame axes), so no extra rotation — but origin must be subtracted both ways.

```python
# Input (W_sim → G):
centroid_t0_W = [pc_W(0).mean()[0:2], TABLE_TOP_Z]   # fixed at episode start
pc_G          = pc_W - centroid_t0_W
ee_pos_G      = ee_pos_W - centroid_t0_W
ee_quat_G     = ee_quat_W                            # W_sim axes ≡ G axes, no rotation

# Policy inference:
action_G = policy(pc_G, ee_pos_G, ee_quat_G)   # action_G in G-frame

# Output (G → W_sim) for RMPFlow execution:
action_pos_W  = action_G[:3] + centroid_t0_W
action_quat_W = action_G[3:7]                  # same axes, no rotation
action_grip   = action_G[7]
franka.RmpFlow(action_pos_W, action_quat_W)
```

Bug to avoid: passing G-frame action directly to RMPFlow without adding back centroid → Franka moves to wrong absolute position.

### 2.4 `baseline1_dataset.py` (add yaw augmentation hook, default OFF)

```python
def __init__(self, ...,
             yaw_augmentation: bool = False,            # Phase 1 default: disabled
             yaw_range: tuple = (-np.pi/6, np.pi/6)):   # ±30° default for Phase 2+
    self.yaw_aug = yaw_augmentation
    self.yaw_lo, self.yaw_hi = yaw_range
    ...

def _sample_to_data(self, sample):
    # ... existing code ...
    if self.yaw_aug:
        yaw = self._rng.uniform(self.yaw_lo, self.yaw_hi)
        data = apply_yaw_rotation(data, yaw=yaw)
    return data
```

**Default**: `yaw_augmentation=False`. Phase 1 uses the default. Phase 2+ enables explicitly with bounded range (±30° / ±60°), NEVER full 0..2π.

`apply_yaw_rotation` helper: ~30 lines, rotates pc / ee_pos / ee_quat / action consistently around +Z.

**Critical: yaw aug applies the SAME yaw to a whole trajectory window, not per-frame**. Per-frame yaw breaks the within-trajectory continuity (state[t] and state[t+1] would be in different yaw rotations → action becomes meaningless).

---

## 3. Phase 1: Single-Object Pilot ⭐ (NEW, per user suggestion)

**Goal**: Get **one** object grasped successfully in sim before scaling to all objects.

### 3.1 Pre-flight validation (~3 h, no GPU)

Numbered gates — each must pass before proceeding.

**Gate 1a: extrinsics convention** ✅ PASSED (May 17)
Confirmed DexYCB extrinsics is `T_W←C` (camera pose in world frame). Cameras agree to < 1mm on object pose in W-frame. **Master cam identity ≠ proof of convention** — must test with non-master cameras as Gate 1a does. See verified plan §2.1 formulas.

**Caveat**: only subject-09 (extrinsics ID `20201014_215638`) is verifiable locally — other 9 subjects' extrinsics files are missing. Phase 1 pilot **restricted to subject-09** until other 9 extrinsics files are downloaded.

**Gate 1b: W→G rotation determination** ✅ PASSED (May 17, corrected via Method B = AprilTag rotation)

**INITIAL ATTEMPT WAS WRONG**: assumed master cam is upright (+Y axis-aligned with gravity). Lift-trajectory heuristic gave `R_x(-90°)` as approximation, which has **37° error** from true gravity direction. Discovered when AprilTag-vs-object positions disagreed by 32cm.

**CORRECT METHOD** (Method B per planned alternatives — turned out AprilTag rotation IS in extrinsics.yml):

AprilTag is physically attached to table surface. Its `[R|t]` in extrinsics file stores `T_W←AprilTag`. AprilTag's local +Z axis points "out of the table surface" = anti-gravity direction.

```python
apriltag_M = np.array(ext['apriltag']).reshape(3, 4)
R_apriltag = apriltag_M[:, :3]   # rotation: apriltag-local → W-frame
table_normal_W = R_apriltag[:, 2]   # = anti-gravity direction (exact)
gravity_W = -table_normal_W

# For DexYCB 20201014_215638 calibration:
# gravity_W = [-0.0172,  +0.7962, +0.6048]  (NOT axis-aligned; 37° from +Y)
# This means master cam is mounted at 37° tilt from upright, looking down-forward at table.

# Build R_W→G: G has +Z = -gravity_W (up), +X = projection of W +X onto horizontal plane
W_x = np.array([1.0, 0.0, 0.0])
W_x_horizontal = W_x - (W_x @ gravity_W) * gravity_W
W_x_horizontal /= np.linalg.norm(W_x_horizontal)
z_G_in_W = -gravity_W
x_G_in_W = W_x_horizontal
y_G_in_W = np.cross(z_G_in_W, x_G_in_W); y_G_in_W /= np.linalg.norm(y_G_in_W)
R_WG = np.column_stack([x_G_in_W, y_G_in_W, z_G_in_W]).T

# Verified result for DexYCB extrinsics_20201014_215638:
R_WG = np.array([
    [+0.9999, +0.0137, +0.0104],
    [ 0.0000, -0.6049, +0.7963],
    [+0.0172, -0.7962, -0.6048],
])
```

**Verification** (all consistent within mm precision):
- `R_WG @ gravity_W = (0, 0, -1)` ✓ (gravity = G -Z)
- apriltag in G: `(0.395, 1.313, -0.688)` → table at z_G = -0.688 (master cam 68.8cm above table)
- can center frame 0 in G: `(0.16, 0.68, -0.613)` = 7.5cm above table (= half can height) ✓

**Lesson learned**: never assume cameras are axis-aligned with world. Use rigorous calibration markers when available; otherwise use Method C (multi-object table-plane fitting).

**OakInk**: must re-verify separately. Check if OakInk's `cam_extr` includes table marker rotation. If not, use Method C.

**Gate 1c: CAD mesh sanity + axis convention**
For ycb_dex_01 (master_chef_can):
- Load `.../models/002_master_chef_can/textured.obj` with trimesh
- Compute bbox: assert height in [10, 16] cm, diameter in [9, 11] cm (real-world dimensions)
- Compute centroid: should be near `(0, 0, 0)` (CAD canonical)
- Verify trimesh loads without errors, surface area is sensible
- **CAD axis convention check** (critical for sim spawn):
  - YCB CAD models follow the YCB benchmark convention. For master_chef_can (a cylinder), the cylinder axis should align with one of the CAD-local axes; document which one (e.g. CAD +Z = cylinder axis = "up" in the natural standing pose)
  - Validate: if you spawn this CAD mesh in IsaacSim with identity orientation, does it stand upright on the table? Or does it lie on its side? Or does it intersect the table?
  - If CAD's "natural up" is NOT IsaacSim's +Z, then `convert_cad_usd.py` must apply a rotation during USD export OR sim spawn must apply orientation correction
  - Document the CAD up axis per object family (YCB: spec says +Z for most; OakInk: TBD)
  - Without this check: CAD baseline can fail in sim even though math is right, because the spawned object lies on its side and gripper can't grasp it

**Gate 1d: CAD projection check**
For one DexYCB frame: render `pose_y_C @ cad_mesh_C` projected to image using camera intrinsics K. Overlay on RGB. Object silhouette should align with visible object (within ~few pixels). This validates that `pose_y_C` is "CAD-frame → camera-frame" as we assume.

**Gate 1e: table_z self-consistency check (per §1.2)** ✅ PASSED (after Gate 1b correction)

Compute table_z_G from object lowest-points across 5 ycb_dex_01 sessions × 6 cameras (= 30 estimates):
- Cross-camera within session: std = 0.0mm (byte-identical) — confirms frame implementation correct
- Cross-session: std = 3mm (vs 10mm threshold) ✅
- vs apriltag-derived table_z_G = -0.6884m: matches within 2.1mm (= mesh bottom precision)

**Note**: initial run of Gate 1e showed std = 58mm because Gate 1b was wrong (37° gravity error). After correcting R_W→G via apriltag rotation, std dropped to 3mm. **Gate 1e PASS is a strong validation of Gate 1b correction.**

Final value: `table_z_G = -0.6884 m` (apriltag-derived) or `-0.6863 m` (object-derived, median across all). Use apriltag-derived since it's the more direct measurement.

**D4 decision update**: keep "object lowest-point" as primary AND cross-check against apriltag — both methods agree to mm precision, give us confidence.

**Gate 2: ⭐ Cross-camera consistency (smoking-gun gate before any training)**

Pick one DexYCB session (e.g., `20200709_141754` with all 6 cameras). For each camera, run retarget through to G-frame using **the same fixed 4096-point sample indices for each mesh** (eliminate trimesh sampling randomness as confound).

```python
# Critical: use shared sample, not re-sample per camera
rng = np.random.default_rng(seed=0)
cad_mesh_pts = trimesh.sample.sample_surface(cad_mesh, n=4096, seed=rng)  # ONCE
shared_face_idx = ...  # reuse for all 6 cameras

for cam in 6_cameras:
    pc_G_cam[t] = retarget(cam, shared_pts=cad_mesh_pts)   # no resampling

# Then compare:
tol_pos  = 0.001   # 1mm (no sampling noise to absorb)
tol_quat = 0.005   # ~0.3° rotation

for t in range(T):
    for c1, c2 in cam_pairs:
        # All 4096 points should match in O-frame (same physical PC)
        pc_diff = np.linalg.norm(pc_G_c1(t) - pc_G_c2(t), axis=-1).max()
        ee_diff = np.linalg.norm(ee_pos_G_c1(t) - ee_pos_G_c2(t))
        quat_diff = quat_angular_distance(ee_quat_G_c1(t), ee_quat_G_c2(t))
        assert pc_diff < tol_pos
        assert ee_diff < tol_pos
        assert quat_diff < tol_quat
```

**What failures mean**:
- All cameras agree (< tol) → ✅ frame implementation correct, safe to proceed
- Cameras disagree by ~constant rotation → extrinsics direction reversed (T_C←W vs T_W←C)
- Cameras disagree by ~constant translation → centroid_t0 / origin_W computation buggy
- Cameras disagree by arbitrary amounts → quat composition wrong, OR W→G rotation wrong

**This single test catches every frame-related bug before we spend 9h training**. Mandatory gate before §3.3.

**Gate 3: GT trajectory replay sanity (smoking-gun for trajectory feasibility)**

Before training DP3, verify the retarget GT trajectory itself is **physically realizable in IsaacSim**. If GT can't be replayed, no policy can learn it.

```python
# For 2-3 episodes:
# 1. Load retarget GT trajectory (state[t], action[t] in G-frame)
# 2. Spawn object in sim at the corresponding initial pose (read from training data)
# 3. Convert each action_G[t] to world frame (add centroid_t0_W)
# 4. Execute via Franka RMPFlow at the same frequency as training
# 5. Compare execution: did Franka reach the targets? Did gripper close at right time?
#    Did object get lifted? Or did Franka collide with table / fail to reach?
```

**What failures mean**:
- GT replay lifts object → ✅ trajectories are sound, DP3 has a fighting chance
- Franka can't reach EE targets → workspace mismatch, retarget EE-OFFSET wrong
- Gripper closes at wrong time → onset detection buggy
- Object falls/rolls instead of being grasped → spawn pose wrong, GT trajectory wrong
- Collisions with table → table_z wrong, action z values wrong

Mandatory gate. **If GT replay fails, no point training DP3** — fix the upstream issue first.

**Gate 4: Initial EE distribution match**

Sim eval starts with Franka at home pose. Training starts with hand at pre-grasp position. These must overlap.

```python
# Compute training state[0] distribution per object (in G-frame):
ee_pos_t0_train = [ep.state[0][:3] for ep in train_eps_ycb01]
ee_quat_t0_train = [ep.state[0][3:7] for ep in train_eps_ycb01]

# Compute sim Franka home pose in G-frame:
ee_home_W = franka.get_home_ee_pose()
ee_home_G = R_WG @ (ee_home_W - centroid_t0_W)

# Check overlap:
distances = [np.linalg.norm(ee_home_G - ep_t0) for ep_t0 in ee_pos_t0_train]
min_dist = min(distances)
assert min_dist < 0.15, f"sim home is {min_dist:.2f}m from any training start — OOD"
```

If FAIL options:
- Move Franka home pose closer to training distribution
- Add "home → approach" bridging segment to training data
- Accept it and trust policy generalization (risky)

### 3.2 Object selection — **ycb_dex_01 (master_chef_can)** recommended

Why:
- Simple cylindrical geometry → policy doesn't need to learn complex shape priors
- Stable on table by gravity (won't roll easily)
- Many training episodes across 10 subjects × 6 cameras ≈ 60+ episodes
- CAD model is high quality (provided by YCB project)

Alternates: ycb_dex_03 (cracker_box, equally easy)

### 3.3 Pilot training run

```bash
# Generate retargeted data for single object only:
python Baseline1/retarget_human_to_ee.py \
    --dataset dexycb \
    --object-filter ycb_dex_01 \
    --output-dir Baseline1/data/episodes_v3_pilot_ycb01

# Convert to zarr:
python Baseline1/convert_to_zarr.py \
    --input_dir Baseline1/data/episodes_v3_pilot_ycb01 \
    --output_zarr Baseline1/data/v3_pilot_ycb01.zarr

# Train (smaller model? fewer epochs?):
WANDB_MODE=offline python train.py \
    --config-name=dp3.yaml task=baseline1 \
    task.dataset.zarr_path=Baseline1/data/v3_pilot_ycb01.zarr \
    training.num_epochs=200 \
    training.checkpoint_every=20 \
    hydra.run.dir=Baseline1/dp3_runs/v3_pilot_ycb01
```

**Expected timing**: 60 episodes × 100 epoch ≈ 1-2 h (much faster than 9h for full dataset)

### 3.4 Pilot success criterion

- **Sim eval**: 10 rollouts of ycb_dex_01, success rate
- **Target**: > 50% success rate (dz > 3cm). Stretch: > 80%
- **Failure analysis**: if < 50%, dump failure mode (which rollouts failed, what dz, what action quality)

### 3.5 Decision after pilot

| Pilot outcome | Next step |
|---|---|
| ✅ > 50% success | Proceed to Phase 2 (multi-object scale-up) |
| ⚠️ 0% < dz < 50% | Debug specific failure (likely Bug C spawn pose) before scaling |
| ❌ 0% success / dz ≈ 0 | Deeper issue — investigate before any expansion. Possibilities: gripper timing, EE retarget bug, yaw augmentation wrong, etc. |

---

## 4. Phase 2: DexYCB Multi-Object Scale-Up

Only enter after Phase 1 (single-object) succeeds (>50% dz>3cm).

### 4.1 Gradual expansion within DexYCB only
1. **Subset: 3 easy objects** (cracker_box, mug, sugar_box — all "stable-on-table" shapes) → train 100 epoch → sim eval
2. **All 20 YCB objects** (~5700 ep) → train 100 epoch → sim eval per object
3. Compare per-object success rates → identify "hard objects" (e.g., asymmetric tools)

### 4.2 Decisions deferred to Phase 2 start (not now)
- Whether to enable yaw augmentation (if Phase 1 worked without it, decide based on Phase 2.1 result)
- Whether to do per-object spawn pose distribution analysis (D8)

### 4.3 Validation
- Loss curves comparable to Phase 1 (no regression from scaling)
- Per-object success rate doesn't drop catastrophically
- Hard objects (low success) are analysable (geometry vs trajectory complexity)

---

## 5. Phase 3: Add OakInk ⏳ (deferred, after DexYCB-all works)

Only enter after Phase 2 (all DexYCB) achieves >50% average success.

### 5.1 Why OakInk now
- 100 more objects → much more variety for cross-object generalization
- Required for comparison with partner main method (which trains on both)
- Different camera setup → tests pipeline robustness

### 5.2 Steps
1. Verify OakInk `cam_extr` convention (deferred check from §3.1.2)
2. Verify OakInk W→G rotation (might differ from DexYCB)
3. Verify CAD quality on 5 OakInk objects (.obj load, surface sample, visual check)
4. Add OakInk path to retarget script
5. Generate combined zarr (DexYCB + OakInk, ~9700 ep)
6. Retrain ~9h
7. Sim eval on held-out objects from both datasets

### 5.3 Risk
OakInk's per-frame `cam_extr` is more complex than DexYCB's static per-session extrinsics. If this introduces bugs, can fall back to "DexYCB-only" Baseline1 and document limitation.

---

## 6. Phase 4: Switch to SAM3D ⏳ (deferred, after Phase 3 succeeds)

**Trigger** (consistent with D8): **Phase 3 (DexYCB + OakInk on CAD) achieves > 50% average success rate**.

Rationale for "after Phase 3" not "after Phase 2":
- Phase 3 validates the OakInk path works on CAD; SAM3D switch will need OakInk too (partner main method covers both)
- If Phase 3 fails (OakInk-specific issues), no point switching to SAM3D — fix OakInk first
- Phase 3 gives complete CAD upper-bound baseline numbers to compare SAM3D drop against

Alternative ("fast apples-to-apples"): start Phase 4 right after Phase 2 (DexYCB-CAD success) with DexYCB-SAM3D only. This is faster (~1 day earlier) but loses OakInk comparability and we'd have to do OakInk separately later. **Default = sequential (after Phase 3)**.

### 6.1 Why switch
- Make Baseline1 comparable to main method (PointNet++ contact prediction also uses SAM3D)
- Match partner's pipeline ecosystem (training_fp, obj_poses, etc.)
- Enable real-world deployment in future (no CAD for novel objects)

### 6.2 Switch steps
1. Run `data/estimate_obj_scale.py --dataset oakink` (Depth Pro, ~10 min) — get partner's scale.json for all OakInk
2. Modify retarget: swap CAD mesh → SAM3D mesh + scale.json
3. Decision: use `R_align` (our R_align JSON) OR `FoundationPose pose` (partner FP output)?
   - **Recommended**: switch to FP pose (consistent with partner main method, no R_align needed)
4. Regenerate zarr, retrain ~9h
5. Sim eval, compare success rate with CAD baseline

### 6.3 Scale consistency rule — apply ONCE, never twice

When swapping CAD → SAM3D, the same mm-vs-m mistake we hit on A16013 (USD was 7× too big) can come back if we accidentally apply `scale_factor` in two places. **Strict rule**:

```
Scale_factor must be applied EXACTLY ONCE in the pipeline. Choose ONE:

Option A (bake into USD at conversion time):
  convert_obj_usd.py reads scale.json, applies × scale_factor to mesh.vertices, exports USD.
  → USD file is in real metres
  → At sim runtime: spawn with scale=1.0
  → At training: load mesh and apply × scale_factor BEFORE sampling (mirrors USD)

Option B (apply at runtime, not in USD):
  convert_obj_usd.py exports SAM3D mesh AS-IS (raw recon units)
  → USD file is in recon units (not metres)
  → At sim runtime: spawn with scale=[scale_factor]*3
  → At training: load mesh and apply × scale_factor BEFORE sampling

NEVER:
  USD already baked with scale_factor + spawn also passes scale_factor → 7× error
  USD raw + spawn scale=1.0 → object too big in sim
  Training uses scale, sim doesn't → train-sim PC dist mismatch
```

For Phase 3 SAM3D switch, **default to Option A** (bake into USD) — it's simpler and the existing partner `convert_obj_usd.py` already does this. Sim eval just spawns scale=1.0, training mirrors by reading mesh and applying scale.json before sampling. Document this in code comments and the dataset README.

For Phase 1 CAD path: scale is always 1.0 (CAD already in metres), this rule is trivially satisfied.

### 6.4 Expected outcome
- SAM3D success rate likely lower than CAD (mesh noise effect)
- Gap = "cost of using imperfect 3D reconstruction"
- Direct apples-to-apples comparison with main method

---

## 7. Decision Points (confirm before execution)

| # | Decision | Default proposal | Need user confirm |
|---|---|---|---|
| D1 | Pilot object | ycb_dex_01 (master_chef_can) | yes |
| D2 | Pilot training epochs | 200 (vs full 100) | yes |
| D3 | yaw augmentation in Phase 1 | **disabled** (Phase 1 uses single object with all 6 cameras for cross-camera validation; if Phase 1 succeeds without yaw aug, defer enabling to Phase 2) | yes |
| D4 | table_z method | **object lowest-point method** (multi-object, multi-frame, median aggregate); AprilTag NOT used (decision: simpler self-contained pipeline) | confirmed |
| D5 | Keep v2 ckpt | yes, for ablation comparison | yes |
| D6 | When to start Phase 2 | After Phase 1 > 50% success | yes |
| D7 | When to start Phase 3 (add OakInk) | After Phase 2 (DexYCB-all) > 50% average | yes |
| D8 | When to start Phase 4 (switch to SAM3D) | After Phase 3 (DexYCB+OakInk CAD) > 50% average | yes |
| D9 | Spawn pose source for sim eval | Phase 1: just "upright on table" (cylinder is trivially upright); Phase 2: extract per-object distribution mode | yes |
| D10 | Phase 4 scale rule (SAM3D) | Option A — bake scale_factor into USD, sim spawn scale=1.0 | yes |
| D11 | Use all 6 DexYCB cameras for training or 1? | Phase 1: **use all 6** (we want Gate 2 cross-cam validation to enforce frame correctness) | yes |

---

## 8. Validation Steps (avoid wasted GPU time)

**Gate sequence** (each gate must pass before proceeding):

### Gate 1 — Pre-retarget (DexYCB conventions)
- ☐ 1a: extrinsics convention (T_C←W vs T_W←C) verified
- ☐ 1b: W→G rotation determined empirically (which Rx sign?)
- ☐ 1c: CAD mesh sanity (master_chef_can bbox ~10cm, loadable, centered)
- ☐ 1d: CAD projection check — `pose_y @ cad_mesh` projects to visible object in image
- ☐ 1e: table_z AprilTag vs object lowest-point delta < 2cm

### Gate 2 — ⭐ Cross-camera consistency (smoking-gun frame test)
- ☐ Same session, 6 cameras, **shared sample indices**, pc_G/ee_G/action_G agree to < 1mm pos / < 0.3° rot
- ☐ If FAIL: do NOT proceed to training — fix extrinsics direction / quat composition / W→G

### Gate 3 — ⭐ GT trajectory replay in sim (smoking-gun feasibility test)
- ☐ Pick 2-3 retargeted episodes, spawn object in sim at training initial pose
- ☐ Execute the GT action sequence via Franka RMPFlow
- ☐ Verify: Franka reaches all targets, gripper timing correct, object lifted, no collisions
- ☐ If FAIL: GT itself isn't physically achievable → fix upstream (EE-OFFSET, gripper onset, table_z) before training

### Gate 4 — Initial EE distribution overlap
- ☐ Franka home pose in G-frame is within 15cm of any training state[0] EE position
- ☐ If FAIL: reset Franka home to closer pose OR add home→approach bridge to training data

### Gate 5 — Before zarr generation
- ☐ Single episode in G-frame visualized (plot pc + EE + gravity arrow), looks physically correct
- ☐ Object Z values at frame 0 are at ground level (near 0 in G-frame)
- ☐ EE Z values during approach are above the object (positive in G-frame)

### Gate 6 — Before full training
- ☐ 10-epoch smoke training: loss monotonically decreases
- ☐ Normalizer fit: input ranges look reasonable (no Inf/NaN)
- ☐ (Phase 2+ if yaw aug enabled): same sample with different yaw seed produces rotated-equivalent (pc, ee, action), magnitudes preserved

### Gate 7 — Before sim eval
- ☐ DP3 server self-test: predict on val sample, MSE vs GT < some threshold
- ☐ Sim PC obs spot-check: shape similar to training PC for same object
- ☐ Sim spawn pose in training distribution (Phase 1: upright; Phase 2: per-object mode)
- ☐ Sim G-frame origin = training convention (centroid_xy + table_z=0.80)
- ☐ Action/control rate aligned (see §10)

---

## 9. Time Estimate

### Phase 1 (DexYCB, single object ycb_dex_01)
| Step | Time | Blocking? |
|---|---|---|
| §3.1 Pre-flight validation | 2 h | no |
| §2.1 Rewrite retarget (DexYCB only) | 2-3 h | no |
| §2.2 CAD→USD converter | 30 min | no |
| §2.3 Update eval_dp3_policy | 30 min | no |
| Pilot retarget + zarr (single obj, ~60 ep) | 30 min | no |
| Pilot train (200 epoch single object) | 1-2 h | yes (GPU) |
| Pilot sim eval | 30 min | yes |
| **Phase 1 total** | **~7-9 h** | mostly compute |

### Phase 2 (DexYCB all 20 objects)
| Step | Time | Blocking? |
|---|---|---|
| Decide yaw aug (based on Phase 1) | trivial | no |
| Full retarget + zarr (~5700 ep) | 2 h | no (CPU parallel) |
| Train 100 epoch | 9 h | yes (GPU) |
| Sim eval per object | 2 h | yes |
| **Phase 2 total** | **~13 h** | half compute |

### Phase 3 (Add OakInk)
| Step | Time | Blocking? |
|---|---|---|
| OakInk convention validation | 1 h | no |
| Extend retarget for OakInk | 2 h | no |
| Full retarget + zarr (~9700 ep) | 4 h | no |
| Train 100 epoch | 9 h | yes |
| Sim eval | 2 h | yes |
| **Phase 3 total** | **~18 h** | half compute |

### Phase 4 (Switch CAD → SAM3D)
| Step | Time | Blocking? |
|---|---|---|
| `estimate_obj_scale.py` for OakInk | 10 min | yes (Depth Pro) |
| Modify retarget to SAM3D + scale.json | 2 h | no |
| Re-generate zarr | 3 h | no |
| Train 100 epoch | 9 h | yes |
| Sim eval comparison vs CAD | 2 h | yes |
| **Phase 4 total** | **~16 h** | half compute |

### Total project budget
~54 h spread over Phases 1-4. Decision gates at each phase boundary — can stop or pivot if any phase fails.

---

## 10. File / Directory Changes

### New files
- `Baseline1/PLAN_V3_CAD_FIRST.md` (this file)
- `Baseline1/tools/convert_cad_usd.py`
- `Baseline1/data/episodes_v3_pilot_ycb01/` (~60 hdf5)
- `Baseline1/data/v3_pilot_ycb01.zarr`
- `Baseline1/dp3_runs/v3_pilot_ycb01/`
- `output/obj_usd_cad/{ycb,oakink}/*.usd`

### Modified files
- `Baseline1/retarget_human_to_ee.py` (major rewrite)
- `sim/eval_dp3_policy.py` (adapt to G-frame + CAD USD)
- `third_party/3D-Diffusion-Policy/.../baseline1_dataset.py` (add yaw aug)

### Unchanged / Reused
- `Baseline1/convert_to_zarr.py` (no changes)
- `Baseline1/eval/dp3_inference_server.py` (no changes)
- `Baseline1/dp3_runs/all_combined_v2/` (kept for v2 vs v3 ablation)

---

## 11. Action / Control Rate Alignment

DP3 outputs `n_action_steps=8` actions per inference. Sim eval executes these as `world.step()`s. The implicit assumption: **1 predicted sub-action = 1 sim physics step**.

But training data has each frame ≈ 1/30s of real time (DexYCB camera framerate). Sim physics defaults to 60 Hz (1/60s per step). So:
- 1 training step ≈ 30 ms real time → policy expects EE to move ~few cm per step
- 1 sim physics step at 60 Hz = 1/60s → only half the time, so EE moves less

**Mismatch consequences**:
- Sim executes too quickly → policy's "next step EE pos" is too far for one physics step → RMPFlow can't keep up → undertracked targets
- OR sim executes too slowly → policy's planned trajectory completes faster than physics → premature termination

**Resolution options**:
1. **Match training rate**: sim runs `n_physics_steps_per_action = round(training_dt / sim_dt) = 2` for 30 Hz training + 60 Hz sim
2. **Match action stride**: control physical execution to take `1/30s` per predicted action regardless of physics rate
3. **Verify in Gate 3 (GT replay)**: if GT replay works at "1 action = 1 step" rate, current setup is OK; otherwise adjust

Current code does `for sub_idx: world.step()` (1 sub-action = 1 step). Verify against Gate 3 replay timing, adjust if Franka under/over-tracks.

---

## 12. Open Questions / Risks

### Q1: OakInk W-frame and W→G rotation
DexYCB W-frame convention is clear (master cam OpenCV). OakInk cam_extr is per-frame, but what's its W-frame? Need to spot-check before retarget.

### Q2: ycb_name mapping for DexYCB
Our SAM3D meshes are `ycb_dex_NN`, CAD meshes are `{ycb_name}` (e.g., `002_master_chef_can`). The mapping is in `obj_meshes/ycb/ycb_dex_NN/scale.json` → `ycb_name` field. Retarget script needs to use this for CAD path lookup.

### Q3: OakInk CAD quality
OakInk CAD models are from real object scans. Some may have:
- Holes / missing geometry → trimesh.sample may produce weird results
- Different topology than what's in our SAM3D meshes (if SAM3D was reconstructed differently)
- Need to verify a few before training

### Q4: yaw augmentation impact on quat sign continuity
Current code does quat sign continuity along trajectory. With yaw aug applied per-sample (not per-trajectory), the within-trajectory continuity may break. Need to verify or apply yaw aug at trajectory level (one yaw per episode).

### Q5: gripper command alignment in G-frame
Current gripper is 0→1 step based on hand-object proximity in C-frame. In G-frame the proximity calculation gives the same result (rigid transform invariant), but verify.

### Q6: spawn pose distribution per object (for D9)
For each object in training data, extract the distribution of `pose_y` rotation at t=0 (the orientation the human placed it in before grasping). This forms the "natural" distribution. Sim spawn should sample from this distribution OR use its mode. For Phase 1 pilot (ycb_dex_01 cylinder), trivial — always upright. For Phase 2, **mandatory** (yaw augmentation alone doesn't solve pitch/roll variations). Defer detailed implementation to start of Phase 2.

### Q7: Project framing — CAD-first is a diagnostic baseline
Baseline1 with CAD is technically not directly comparable to the main method (which uses SAM3D). This plan's deliverable should be framed as:

- **Phase 1-3 deliverable**: "Baseline1-CAD" — a DP3 trajectory imitation baseline trained on CAD geometry + GT pose. This is an **upper-bound / oracle-geometry** baseline that establishes whether DP3 can learn this task at all.
- **Phase 4 deliverable**: "Baseline1-SAM3D" — same method but with realistic mesh source. This is the **apples-to-apples** baseline for comparison with main method.

Document clearly in any report/paper that CAD-baseline numbers represent "best case with perfect geometry", not "what we'd see in deployment".

---

## 13. Status

- [ ] Plan reviewed and confirmed by user + partner
- [ ] Phase 1 (DexYCB ycb_dex_01) pilot started
- [ ] Phase 1 pilot succeeded (> 50% success rate)
- [ ] Phase 2 (DexYCB all 20 objects) started
- [ ] Phase 2 succeeded
- [ ] Phase 3 (add OakInk) started
- [ ] Phase 3 succeeded
- [ ] Phase 4 (switch to SAM3D) started
- [ ] Phase 4 succeeded → final Baseline1 ready

> **Owner**: claude + fanxu
> **Last updated**: 2026-05-17

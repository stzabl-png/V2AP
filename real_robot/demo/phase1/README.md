# Phase 1 — Scripted pick-and-lift demo

Phase 1 is a **filmed real-robot demo**: for each object, a YAML stores arm joints and a derived `R_ee` grasp pose; the **right Sharpa HA4** acts as a **thumb+index virtual gripper** (middle/ring/pinky splayed for clearance). Pre-grasp and lift follow the UCB sim convention (**0.15 m** retreat along `-approach`, **+0.15 m** world Z lift).

**Upstream context:** affordance / mesh pipeline lives in [UCB_Project (titan)](https://github.com/stzabl-png/UCB_Project/tree/titan). Phase 2 will automate grasp selection; Phase 1 is **manual tuning on hardware**.

See also: [repo README](../../README.md) · [demo overview](../README.md)

---

## What you need

| Item | Notes |
|------|--------|
| **Machine** | Lab Razor laptop (Ubuntu), on robot subnet |
| **Env** | `source setup_local.sh` (or `setup.sh`); conda env with dexcontrol + SharpaWaveSDK |
| **Extra pip** | `pip install pyyaml` |
| **Zenoh** | `~/.dexmate/comm/zenoh/` copied from a working lab machine |
| **Robot** | Vega powered, Zenoh listener up (`ROBOT_IP` reachable on `:7447`) |
| **Hands** | Both Sharpa HA4 connected; serials via env if needed (below) |
| **Safety** | Physical e-stop within reach; no dedicated e-stop key in scripts |

```bash
# Sanity check before tuning
python teleop/reset_arm_to_init_pose.py
```

---

## Repository layout (Phase 1)

```text
demo/
├── right_hand_profile.yaml     # shared open/closed hand (all objects)
├── hand_tuner.py               # tune hand profile once
├── hand_close.py               # stall-detecting pinch close
├── virtual_gripper.py          # default hand preset if profile is null
├── hardware.py                 # robot + hand connect
└── phase1/
    ├── configs/
    │   ├── start.yaml          # demo start pose (required for full sequence)
    │   ├── <object>.yaml       # per-object grasp (e.g. chips.yaml)
    │   └── _template.yaml      # schema reference
    ├── pose_tuner.py           # tune per-object right arm + preview motion
    ├── run_grasp.py            # filmed demo entry point
    ├── executor.py             # motion sequence + planning
    ├── config_io.py            # YAML load/save
    ├── grasp_geometry.py       # pre-grasp / lift math
    └── test_stall_close.py     # hand close smoke test
```

**Design split**

| What | Where it lives |
|------|----------------|
| **Hand open/closed** (22-DOF, shared) | `demo/right_hand_profile.yaml` via `hand_tuner.py` |
| **Arm grasp** (right arm joints + FK `grasp_pose`) | `demo/phase1/configs/<object>.yaml` via `pose_tuner.py` |
| **Demo start pose** | `demo/phase1/configs/start.yaml` |

Object YAMLs use `hand_open_joint_pos: null` / `hand_closed_joint_pos: null` → profile is loaded automatically.

---

## End-to-end workflow (first time on a new setup)

```text
┌─────────────────────────────────────────────────────────────────┐
│ 0. Lab setup: clone repo, conda env, Zenoh, source setup.sh     │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Hand profile (ONCE per hand / lab)                           │
│    python demo/hand_tuner.py  →  demo/right_hand_profile.yaml   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Start pose (ONCE)                                            │
│    Manually move robot to filming “idle” pose                   │
│    python demo/phase1/pose_tuner.py --object-name start  →  >   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Per object (repeat for chips, mug, …)                        │
│    Place object at fixed table spot                             │
│    python demo/phase1/pose_tuner.py --object-name chips  →  >   │
│    X = full sequence test; tune until pick works                │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Film demo                                                    │
│    python demo/phase1/run_grasp.py --object-name chips          │
└─────────────────────────────────────────────────────────────────┘
```

Mark the object location on the table — configs assume **the same placement every run**.

---

## Step 1 — Tune hand profile (`hand_tuner.py`)

Tunes **only the right hand** (arm does not move). Saves **`demo/right_hand_profile.yaml`**, used by all objects.

**Convention** (built into `virtual_gripper.py` defaults; refine on hardware):

- Middle / ring / pinky: straight, MCP splayed for clearance (same in open and closed).
- Thumb + index: straight phalanges; **open** = roots spread; **closed** = roots pinch (stall close stops on contact).

```bash
python demo/hand_tuner.py
```

| Key | Action |
|-----|--------|
| `1`–`5` | Select finger (thumb … pinky) |
| `J` / `K` | Next / prev joint in finger |
| `W` / `S` | Nudge selected joint ± |
| `O` | Copy **live** hand → **open** slot (auto-saves profile) |
| `C` | Copy **live** hand → **closed** slot (auto-saves profile) |
| `[` / `]` | Apply open/closed **slots** from YAML to hand |
| `G` / `N` | Apply open / closed slot to hardware |
| `>` | Save open/closed slots to disk |
| `Esc` | Quit |

**Tip:** After tuning with `W`/`S`, press **`C`** (and **`O`** for open) before relying on stall-close in the demo — disk must match what you see.

Optional one-shot apply:

```bash
python demo/hand_tuner.py --apply-open
python demo/hand_tuner.py --apply-closed
```

Override profile path: `export V2AP_HAND_PROFILE=/path/to/profile.yaml`

---

## Step 2 — Tune start pose (`pose_tuner.py --object-name start`)

Creates or edits **`configs/start.yaml`**: left arm, torso, head, right arm at the **demo idle** configuration. Required for the full `run_grasp` teardown (return to start after filming).

```bash
python demo/phase1/pose_tuner.py --object-name start
```

**If `start.yaml` does not exist:** script **captures current robot state without moving** — position the robot first, then tune and **`>`** save.

**If `start.yaml` exists:** loads file, opens grip, moves arms to saved grasp joints.

---

## Step 3 — Tune each object (`pose_tuner.py --object-name <object>`)

Creates or edits **`configs/<object>.yaml`**. Tuning is **right-arm joint space** (7-DOF), applied **live** on each `W`/`S`. Saved `grasp_pose` is **FK from joints** on **`>`** (not edited directly in the tuner).

```bash
python demo/phase1/pose_tuner.py --object-name chips
# Match your table for collision planning:
python demo/phase1/pose_tuner.py --object-name chips --table-height 0.76
```

| Key | Action |
|-----|--------|
| `1`–`7` | Select `R_arm_j1` … `R_arm_j7` |
| `W` / `S` | Nudge selected joint (live arm update) |
| `R` | Read current `right_arm` from hardware |
| `[` | Apply profile **open** to hand |
| `]` | Preview **stall close** → hold → reopen |
| `N` | Move to **pre-grasp** (slow IK + table collision) |
| `G` | Return to saved **grasp joints** (slow planned move) |
| `X` | Run full pick sequence (uses `start.yaml` if present) |
| `>` | Save YAML (+ refresh `grasp_pose` from FK) |
| `<` | Reload YAML from disk (does not move arm) |
| `Esc` | Quit |

Hand joints are **not** edited here — use `hand_tuner.py`.

---

## Step 4 — Run filmed demo (`run_grasp.py`)

```bash
python demo/phase1/run_grasp.py --object-name chips
```

Loads `configs/chips.yaml` and `configs/start.yaml` (default `--start-object-name start`).

### Motion sequence

1. **Start pose** — open grip, planned move to `start.yaml` joints  
2. **Enter** — operator places object on table *(background plans pre-grasp while waiting)*  
3. **Pre-grasp** — IK to pose 15 cm back along approach; hand open  
4. **Grasp** — move to tuned right-arm joints; hand still open  
5. **Close** — stall-detecting pinch until contact or target  
6. **Lift** — +15 cm world Z; closed hand  
7. **Hold** `hold_time_s` (default 3 s) — **film here**  
8. **Enter** — move back down to grasp pose (hand closed)  
9. Open grip → **Enter** — remove object  
10. Return to **start** (unless `--skip-home-at-end`)

### Filming variant (stay at lift)

```bash
python demo/phase1/run_grasp.py --object-name chips --skip-home-at-end
```

Skips steps 8–10; arm stays at lift after the hold.

### Other flags

| Flag | Purpose |
|------|---------|
| `--dry-run` | IK / plan logging only; no hardware |
| `--config-dir PATH` | Alternate configs directory |
| `--start-object-name NAME` | Different start yaml (default `start`) |

---

## Config file reference (`configs/<object>.yaml`)

| Field | Purpose |
|-------|---------|
| `object_name` | Label in prompts |
| `table_height` | Table collision box for OMPL (default **0.98 m**; use measured height) |
| `grasp_pose` | 4×4 `R_ee` in base frame (updated from FK on save in pose_tuner) |
| `right_arm_joint_pos` | **Primary grasp tune target** (7 floats) |
| `left_arm_joint_pos`, `torso_joint_pos`, `head_joint_pos` | Fixed during object grasp |
| `pre_grasp_pose` / `lift_pose` | Usually `null` (auto from `grasp_pose`) |
| `pre_grasp_offset_m` / `lift_height_m` | Default **0.15** each |
| `hold_time_s` | Pause at lift for camera (default **3**) |
| `hand_open_joint_pos` / `hand_closed_joint_pos` | Usually **`null`** → `right_hand_profile.yaml` |

Copy `_template.yaml` when adding a new object name.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ROBOT_NAME`, `ROBOT_IP` | Set by `setup.sh` |
| `V2AP_LEFT_HAND_SERIAL` | Left Sharpa serial (optional) |
| `V2AP_RIGHT_HAND_SERIAL` | Right Sharpa serial (optional) |
| `V2AP_HAND_PROFILE` | Override hand profile YAML path |

---

## Off-robot checks (Mac / CI)

No robot required:

```bash
python demo/phase1/grasp_geometry.py
python demo/phase1/run_grasp.py --object-name chips --dry-run
python demo/phase1/pose_tuner.py --object-name chips --dry-run
python demo/hand_tuner.py --dry-run
```

---

## Utilities

```bash
# Stall-close smoke test (right hand only)
python demo/phase1/test_stall_close.py
```

---

## Geometry conventions (aligned with UCB sim)

From `grasp_pose` (4×4 homogeneous, **`R_ee`** in robot base):

- **Approach direction** = 3rd column of rotation (`R[:, 2]`)
- **Pre-grasp** = grasp translation − **0.15 m × approach**
- **Lift** = grasp translation + **0.15 m** in world **+Z**

Head during demo: **`head_j1` pitched down 20°** (`HEAD_PITCH_DOWN_DEG` in `demo/phase1/constants.py`), applied on every `run_grasp` regardless of YAML.

Planned arm moves use **0.3 rad/s** joint speed (`PLANNED_MOTION_JOINT_SPEED_RAD_S`).

---

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| Zenoh connection failed | `ping $ROBOT_IP`, port 7447, robot stack running, `~/.dexmate/comm/zenoh/` |
| Planner fails on pre-grasp | `table_height` in object yaml; try small joint tweaks or `N` in pose_tuner |
| Pinch slips or crushes | Re-tune `hand_tuner.py`; test with `]` in pose_tuner or `test_stall_close.py` |
| Object miss at grasp | Re-tune `pose_tuner.py`; verify object placement matches tuning |
| No return to start | Create `start.yaml`; ensure `run_grasp` finds it |
| Hand wrong but arm OK | Object yaml should have `hand_*: null`; edit profile not per-object hand fields |

---

## Quick command cheat sheet

```bash
source setup_local.sh

# Once
python demo/hand_tuner.py
python demo/phase1/pose_tuner.py --object-name start    # > to save

# Per object
python demo/phase1/pose_tuner.py --object-name chips    # > to save, X to test
python demo/phase1/run_grasp.py --object-name chips
python demo/phase1/run_grasp.py --object-name chips --skip-home-at-end
```

---

## Phase 2 (not implemented here)

Automatic grasp from mesh + affordance inference, hand-eye calibration, and selection without per-object joint tuning. Phase 1 configs remain useful as baselines and filming fallbacks.

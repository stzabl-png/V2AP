# V2AP Real Robot Deployment

Real-robot deployment code for **V2AP** using Dexmate Vega arms with Sharpa HA4 dexterous hands.

## Hardware Requirements

| Component | Spec |
|-----------|------|
| Robot | Dexmate Vega arm |
| Hand | SharpaWave HA4 (5-finger, tactile) |
| Camera | ZED (head-mounted, RGB-D) |
| OS | Ubuntu 20.04 / 22.04 / 24.04, x86_64 |
| Python | 3.10+ (conda recommended) |
| Network | Robot subnet reachable (`ROBOT_IP:7447`) |

## External Dependencies (install separately)

```bash
# Dexmate control stack (AGPL-3.0 / commercial)
bash scripts/install_dexmate.sh

# SharpaWave SDK (proprietary — contact Sharpa)
bash scripts/install_sharpa.sh
```

## Setup

```bash
# 1. Copy and edit the setup template
cp setup.sh setup_local.sh    # fill in ROBOT_NAME, ROBOT_IP
source setup_local.sh

# 2. Copy and edit server connection config (Phase 2 only)
cp demo/phase2/server_client_env.example server_client_env.sh
source server_client_env.sh

# 3. Verify robot connection
python teleop/reset_arm_to_init_pose.py
```

## Phase 1 — Scripted Grasps

Hard-coded grasp poses per object. Uses Sharpa right hand as a virtual 2-finger gripper.

```bash
# Calibrate hand open/close
python demo/hand_tuner.py

# Tune grasp pose for each object
python demo/phase1/pose_tuner.py --object-name chips_can

# Run demo
python demo/phase1/run_grasp.py --object-name chips_can

# Dry-run (no robot required)
python demo/phase1/run_grasp.py --object-name chips_can --dry-run
```

See `demo/phase1/README.md` for the complete filming workflow.

## Phase 2 — Affordance-Driven Automation

Connects to the V2AP pipeline server (GPU machine running `model/inference`) for automatic grasp pose generation.

```bash
# Capture a session (RGB-D + robot state)
python demo/phase2/capture_session.py

# Run automated grasp (pipeline server must be running on GPU machine)
python demo/phase2/run_auto_grasp.py
```

See `demo/phase2/README.md` and `demo/phase2/SERVER_CLIENT_PLAN.md`.

## Environment Variables

| Variable | Purpose | Set in |
|----------|---------|--------|
| `ROBOT_NAME` | Dexmate robot config ID | `setup_local.sh` |
| `ROBOT_IP` | Robot control IP | `setup_local.sh` |
| `CAMERA_IP` | ZED camera host IP | `setup_local.sh` |
| `PIPELINE_TITAN_ROOT` | V2AP root on GPU server | `server_client_env.sh` |
| `PIPELINE_TITAN_SSH_HOST` | GPU server SSH address | `server_client_env.sh` |

## License Notes

- `demo/`, `camera/` — MIT (V2AP authors)
- `dexmate/dexcontrol` — AGPL-3.0 / Dexmate commercial (install separately)
- `SharpaWaveSDK` — Sharpa proprietary (install separately)

Zenoh peer config: copy `~/.dexmate/comm/zenoh/` from your lab machine or Dexmate docs.

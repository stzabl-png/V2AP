# V2AP-demo

Real-robot demo infrastructure for **Dexmate Vega** arms with **Sharpa HA4** dexterous hands. This repo is forked from the lab teleop stack and will host V2AP pick-and-lift demos under `demo/` (Phase 1 scripted grasps, Phase 2 affordance-driven automation).

Upstream affordance / grasp planning code lives in [UCB_Project (titan)](https://github.com/stzabl-png/UCB_Project/tree/titan).

## Requirements

- **OS:** Ubuntu 20.04 / 22.04 / 24.04, x86_64
- **Python:** 3.10+ (conda recommended)
- **Network:** Lab robot subnet (see environment variables below)
- **Hardware:** Dexmate Vega, SharpaWave hands, optional ZED camera streams

## Quick start (lab machine)

```bash
git clone https://github.com/<your-username>/V2AP-demo.git
cd V2AP-demo

# Create or clone the lab conda env, then install local packages:
# pip install -e dexmate/dexcontrol dexmate/dexbot-utils
# pip install -e SharpaWaveSDK

cp setup.sh setup_local.sh   # edit conda env name if needed; setup_local.sh is gitignored
source setup_local.sh

# Sanity check (robot must be idle and powered on):
python teleop/reset_arm_to_init_pose.py
```

## Environment

`setup.sh` sets:

| Variable | Purpose |
|----------|---------|
| `ROBOT_NAME` | Dexmate robot config id (e.g. `dm/vgd1262ab823-1p`) |
| `ROBOT_IP` | Robot control IP on the lab network (e.g. `192.168.50.20`) |

Dexcontrol also expects Zenoh peer config under `~/.dexmate/comm/zenoh/` (copy from lab docs or an existing working machine).

Hand serial numbers are configured in teleop scripts (e.g. `teleop/main_teleop.py`); update for your hand pairing when needed.

## Repository layout

```text
V2AP-demo/
├── teleop/           # Original lab teleop, eval, and arm/hand control
├── dexmate/          # Vendored dexcontrol + dexbot-utils
├── SharpaWaveSDK/    # Sharpa hand SDK
├── camera/           # Head / wrist camera receivers
├── manus/            # Manus glove retargeting (full teleop only)
├── vive_tracker/     # Vive streaming (full teleop only)
├── demo/             # V2AP demo (Phase 1 scripted grasps; Phase 2 planned)
└── setup.sh          # Conda env + robot env vars
```

## Full teleop (optional)

The original bimanual teleop pipeline:

1. Start camera streamers on lab hosts (head / wrist).
2. Manus client + retargeting (`manus/`).
3. `python teleop/teleop_pub.py` then `python teleop/main_teleop.py --data-dir <path>`.

See `teleop_launcher.sh` for the multi-tab launcher. Lab-specific credentials and hostnames are kept in **`README.original.md`** (gitignored; not uploaded to GitHub).

**Manus SDK:** `manus/SharpaManusLinuxClient/ManusSDK/lib/*.so` exceeds GitHub’s file size limit and is gitignored. Copy those binaries from the lab machine if you need full Manus teleop.

## V2AP demo

| Phase | Description |
|-------|-------------|
| **Phase 1** | Hard-coded grasp poses per object; Sharpa right hand as thumb+index virtual gripper; film demo — **[full guide](demo/phase1/README.md)** |
| **Phase 2** | Mesh + affordance inference, hand-eye calibration, automatic grasp and lift |

**Phase 1** — see **[`demo/phase1/README.md`](demo/phase1/README.md)** for the complete setup and filming workflow.

Quick commands (Razor, robot connected):

```bash
pip install pyyaml   # if not already in conda env
source setup_local.sh

python demo/hand_tuner.py                              # once: hand open/closed
python demo/phase1/pose_tuner.py --object-name start   # once: demo start pose
python demo/phase1/pose_tuner.py --object-name chips   # per object: tune grasp
python demo/phase1/run_grasp.py --object-name chips      # film demo
```

Off-robot: `python demo/phase1/grasp_geometry.py` · `run_grasp.py --dry-run`

Sim pre-grasp convention (from UCB `sim/run_grasp_sim.py`): retreat **15 cm** along `-approach_dir` from the grasp pose (`approach_dir` = third column of grasp rotation).

## License notes

- `dexmate/dexcontrol` — Dexmate dual license (AGPL-3.0 / commercial)
- `SharpaWaveSDK` — follow Sharpa license terms before redistribution

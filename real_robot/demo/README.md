# V2AP demo

Shared hand tooling plus **Phase 1** scripted pick-and-lift on the Dexmate Vega + Sharpa HA4, and **Phase 2** affordance grasp via Titan GPU server.

| Doc | Contents |
|-----|----------|
| **[Phase 2 pipeline (Razor ↔ Titan)](PHASE2_PIPELINE.md)** | **Full pipeline: setup, commands, step-by-step** |
| **[Phase 1 guide](phase1/README.md)** | Hand profile → start pose → per-object grasp → film demo |
| **[Phase 2 spec](phase2/README.md)** | Session I/O schemas, retarget, IK, capture details |
| [Repo README](../README.md) | Clone, env, Zenoh, teleop stack |

## Dependencies

```bash
pip install pyyaml matplotlib
source setup.sh   # lab conda env on Razor
```

## Layout

```text
demo/
├── PHASE2_PIPELINE.md        # Razor ↔ Titan runbook (start here for Phase 2)
├── constants.py
├── hardware.py
├── hand_close.py
├── hand_tuner.py
├── right_hand_profile.yaml
├── phase1/                   # see phase1/README.md
│   ├── configs/
│   ├── pose_tuner.py
│   └── run_grasp.py
└── phase2/                   # see phase2/README.md + PHASE2_PIPELINE.md
    ├── run_server_client_pipeline.py
    ├── capture_session.py
    ├── run_auto_grasp.py
    ├── server_client_env.example
    └── sessions/             # gitignored
```

## Quick start

**Phase 1 (manual grasp YAML):**

```bash
python demo/hand_tuner.py
python demo/phase1/pose_tuner.py --object-name start
python demo/phase1/pose_tuner.py --object-name chips
python demo/phase1/run_grasp.py --object-name chips
```

**Phase 2 (Titan affordance + auto grasp):**

```bash
source setup.sh
source demo/phase2/server_client_env.example

python demo/phase2/run_server_client_pipeline.py \
  --object-name chips \
  --capture-extra --sam-point 320 180
```

Full setup, Titan-side commands, and pipeline steps: **[PHASE2_PIPELINE.md](PHASE2_PIPELINE.md)**.

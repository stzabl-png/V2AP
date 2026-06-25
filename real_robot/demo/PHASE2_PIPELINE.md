# Phase 2 — Razor ↔ Titan demo pipeline

Perception on **Titan** (GPU server), execution on **Razor** (Dexmate Vega + Sharpa HA4).

| Machine | Role |
|---------|------|
| **Razor** | Capture RGB-D + calibration → upload session → download results → IK + grasp |
| **Titan** | SAM → SAM3D → metric scale → FoundationPose → PDM grasp candidates |

**Detailed I/O schemas:** [phase2/README.md](phase2/README.md), [phase2/TITAN_OUTPUT.md](phase2/TITAN_OUTPUT.md)  
**Titan server code:** [UCB_Project `titan` branch](https://github.com/stzabl-png/UCB_Project/tree/titan/demo)

---

## Pipeline overview

```text
Razor                                              Titan (segment_daemon)
─────                                              ─────────────────────

 R1  Place object, robot at start
 R2  capture_session.py  →  demo/phase2/sessions/<id>/input/
 R3  rsync upload input/
 ───────────────────────────────────────────────►  demo/sessions/<id>/input/

 R3b ssh mark input/.upload_complete
 ───────────────────────────────────────────────►  daemon picks up session

                                                   T2  SAM2 web UI (Flask :7860)
                                                       operator via SSH tunnel
                                                       Save mask → Done
                                                   T3–T7  batch pipeline
                                                   status.json success=true

 R4  poll status.json + daemon_state.json
 R5  rsync download output/
 ◄───────────────────────────────────────────────  demo/sessions/<id>/output/

 R6  Review T3–T6 PNGs (blocking popups)
 R7  run_auto_grasp.py → open-grip IK → OMPL → stall-close → lift
 R8  Review selected grasp preview (blocking popup) → execute motion
```

**SAM2 during T2:** Titan runs Flask on `127.0.0.1:7860`. The Razor orchestrator **automatically** starts a background `ssh -L` tunnel and opens the browser when Titan enters `waiting_segment`. No separate terminal needed.

To disable (headless / CI): `--no-sam-popup` or `PIPELINE_TITAN_AUTO_SAM_POPUP=0`.

**Session ID:** `YYYYMMDD_HHMMSS_<object_slug>` (e.g. `20260602_192346_chips`)

| Side | Session root |
|------|----------------|
| Razor | `$RAZOR_REPO/demo/phase2/sessions/<session_id>/` |
| Titan | `$TITAN_ROOT/demo/sessions/<session_id>/` |

Each session has `input/` (Razor writes) and `output/` (Titan writes).

---

## One-time setup

### Razor

1. **Repo** — clone/sync this repo (e.g. `~/V2AP-demo`).
2. **Conda + robot** — `source setup.sh` from repo root.
3. **Phase 1 start pose** — once per lab setup:
   ```bash
   python demo/phase1/pose_tuner.py --object-name start
   ```
4. **SSH to Titan** — passwordless login:
   ```bash
   ssh vision@<your-gpu-server> 'echo OK'
   ```
   Optional `~/.ssh/config` host alias `titan-pipeline`.
5. **Pipeline env** — edit and source:
   ```bash
   source demo/phase2/server_client_env.example
   ```
6. **Extra Python** — `pip install matplotlib` (blocking image review windows).
7. **ZED stream** — on dexmate-nano, start `zed_streamer` with depth before capture.

### Titan

1. **Repo** — checkout [`titan` branch](https://github.com/stzabl-png/UCB_Project/tree/titan) at e.g. `/path/to/V2AP`.
2. **Session directory:**
   ```bash
   export TITAN_ROOT=/path/to/V2AP
   mkdir -p "$TITAN_ROOT/demo/sessions"
   ```
3. **Conda envs** — `bundlesdf` (T4/T5/T6/T7), `sam3d-objects` (T3; orchestrator switches automatically).
4. **FoundationPose:**
   ```bash
   export FP_ROOT="$TITAN_ROOT/third_party/FoundationPose"
   ```
5. **SSH** — add Razor public key to Titan `~/.ssh/authorized_keys`.
6. **Start segment daemon** (once per Titan boot / lab session):
   ```bash
   cd "$TITAN_ROOT"
   export FP_ROOT="$TITAN_ROOT/third_party/FoundationPose"
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate bundlesdf

   python -m demo.pipeline.segment_daemon
   ```
   Daemon watches `$TITAN_ROOT/demo/sessions/` for `input/.upload_complete`, opens SAM2 web UI when needed, then runs T3–T7.

7. **Legacy smoke test** (batch T2 without daemon — needs `prompt.json` or existing mask):
   ```bash
   python -m demo.pipeline.process_razor_session \
     --session-dir demo/sessions/<session_id> \
     --device cuda
   ```

---

## Environment variables (Razor orchestrator)

Set in `demo/phase2/server_client_env.example` or `~/.bashrc`:

| Variable | Example | Meaning |
|----------|---------|---------|
| `PIPELINE_TITAN_ROOT` | `/path/to/V2AP` | Titan repo root (`$TITAN_ROOT`) |
| `PIPELINE_TITAN_SSH_HOST` | `vision@<your-gpu-server>` | SSH target |
| `PIPELINE_TITAN_MODE` | `daemon` | `daemon` (default) or `ssh_pipeline` (legacy) |
| `PIPELINE_TITAN_SEGMENT_TUNNEL_PORT` | `7860` | Local port for SAM2 SSH tunnel |
| `PIPELINE_TITAN_AUTO_SAM_POPUP` | `1` | Auto tunnel + browser on `waiting_segment` |
| `PIPELINE_REMOTE_SESSIONS_SUBDIR` | `demo/sessions` | Under `$TITAN_ROOT` on Titan |
| `PIPELINE_TITAN_CMD` | see below | Legacy remote pipeline command |
| `PIPELINE_TITAN_CONDA_ENV` | `bundlesdf` | Conda env for remote T4–T7 |
| `PIPELINE_RAZOR_REPO` | `$HOME/V2AP-demo` | This repo on Razor |

```bash
export PIPELINE_TITAN_ROOT=/path/to/V2AP
export PIPELINE_TITAN_SSH_HOST=vision@<your-gpu-server>
export PIPELINE_TITAN_MODE=daemon
export PIPELINE_TITAN_SEGMENT_TUNNEL_PORT=7860
export PIPELINE_REMOTE_SESSIONS_SUBDIR=demo/sessions
export PIPELINE_TITAN_CMD='python -m demo.pipeline.process_razor_session --session-dir {session_dir} --device cuda'
```

`{session_dir}` expands to `demo/sessions/<session_id>` relative to `$TITAN_ROOT`.

---

## Commands — Razor

All commands run on the **Razor laptop** from repo root after `source setup.sh`.

### Full automated pipeline (recommended)

Object on table, robot powered, ZED streaming. **Titan `segment_daemon` must be running.**

```bash
source setup.sh
source demo/phase2/server_client_env.example

python demo/phase2/run_server_client_pipeline.py --object-name chips
```

When Titan enters `waiting_segment`, the orchestrator **auto-starts SSH tunnel + opens browser** to http://127.0.0.1:7860. Click object → **Save mask** → **Done**. Titan continues T3–T7 automatically.

Manual fallback: `--no-sam-popup`, then run `ssh -L 7860:127.0.0.1:7860 $PIPELINE_TITAN_SSH_HOST` yourself.

- After download: **T3 → T4 → T5 → T6** PNGs pop up sequentially; close each window to continue.
- Before motion: **grasp preview** popup for the selected IK candidate; close window to start execution.

**Skip review popups (lab automation):**

```bash
python demo/phase2/run_server_client_pipeline.py \
  --object-name chips \
  --no-titan-vis \
  --grasp-extra --debug
```

**Legacy batch pipeline** (no daemon; needs `--sam-point` or `prompt.json`):

```bash
python demo/phase2/run_server_client_pipeline.py \
  --object-name chips \
  --ssh-pipeline \
  --capture-extra --sam-point 320 180
```

### Multi-grasp testing (one perceive, N grasps)

```bash
python demo/phase2/run_server_client_pipeline.py \
  --object-name chips \
  --grasp-attempts 5 \
  --capture-extra --sam-point 320 180 \
  --no-titan-vis
```

**Flag order:** put `--grasp-attempts` **before** `--capture-extra` / `--grasp-extra` (or rely on auto-promotion if you forget). Wrong: `--capture-extra … --grasp-attempts 5` used to silently stay at 1 attempt.

- **`--titan-max-candidates`** — Razor writes `pipeline.titan.max_candidates` in `input/session.json` before upload. **Titan T6 must read this** and run PDM with `--n-samples` (default 50).
- **`--grasp-attempts N`** (default 1) — run `run_auto_grasp` **N times** after one perceive. **Default:** each run searches **all** Titan candidates exported in the session (shuffle + IK fallback) until **pre_grasp + grasp IK + table checks** pass, then executes that grasp. **N runs ⇒ N IK-feasible grasps** (may reuse the same rank if it keeps winning the pool). Cap the IK pool with `--grasp-extra --max-candidates N` if needed.
- **Grasp preview** — before each attempt’s motion, a **blocking** T6-style grasp preview window opens (close window to continue). Motion still uses `--no-prompts` (no Enter between phases). Skip with `--grasp-extra --no-visualize`.
- **`--no-candidate-fallback`** — multi-grasp **test** mode: attempt `i` passes `--attempt-index i` (fixed sorted rank only). IK fail on that rank **does not** try other candidates in the same run; pipeline **continues** to attempt `i+1`. Same flag on standalone `run_auto_grasp.py`.

### Step-by-step (manual debug)

**R2 — Capture only**

```bash
python demo/phase2/capture_session.py \
  --object-name chips \
  --sam-point 320 180
```

**R3 — Upload input to Titan**

```bash
SESSION=<session_id>
rsync -avz --progress \
  demo/phase2/sessions/${SESSION}/input/ \
  ${PIPELINE_TITAN_SSH_HOST}:${PIPELINE_TITAN_ROOT}/demo/sessions/${SESSION}/input/
```

**R7 — Grasp only** (after `output/` is on Razor)

```bash
python demo/phase2/run_auto_grasp.py --session-id <session_id>
python demo/phase2/run_auto_grasp.py --session-id <session_id> --debug   # no Enter prompts
```

### Partial pipeline flags

```bash
# Re-use existing capture; run Titan + grasp
python demo/phase2/run_server_client_pipeline.py \
  --session-id 20260602_192346_chips --skip-capture

# Transport only (no robot motion)
python demo/phase2/run_server_client_pipeline.py \
  --session-id 20260602_192346_chips \
  --skip-capture --skip-grasp

# Dry-run transport paths
python demo/phase2/run_server_client_pipeline.py \
  --object-name chips --dry-run-transport
```

---

## Commands — Titan

Run on the **Titan GPU server** inside `$TITAN_ROOT`.

### segment_daemon (default — start once)

```bash
export TITAN_ROOT=/path/to/V2AP
cd "$TITAN_ROOT"
export FP_ROOT="$TITAN_ROOT/third_party/FoundationPose"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bundlesdf

python -m demo.pipeline.segment_daemon
```

Poll from Razor: `output/status.json` (`state`: `waiting_segment` → `running` → done) and `output/daemon_state.json` (`segment_url`).

### Full pipeline (legacy — one session, batch T2)

```bash
python -m demo.pipeline.process_razor_session \
  --session-dir demo/sessions/<session_id> \
  --device cuda
```

Equivalent entry:

```bash
python -m demo.pipeline --session-dir demo/sessions/<session_id>
```

**Useful flags:**

| Flag | When |
|------|------|
| `--skip-sam` | `output/segment/mask.png` already exists |
| `--skip-sam3d` | `output/mesh/object_raw.glb` already exists |
| `--skip-fp` | `output/register/T_cam_mesh.json` already exists |
| `--redo` | Overwrite existing step outputs |

### T7 only (re-finalize status after manual fixes)

```bash
python demo/scripts/T7/write_status.py \
  --session-dir demo/sessions/<session_id> \
  --pipeline-version demo.pipeline.process_razor_session
```

### Download for Razor (manual rsync, on Razor)

```bash
SESSION=<session_id>
rsync -avz --progress \
  ${PIPELINE_TITAN_SSH_HOST}:${PIPELINE_TITAN_ROOT}/demo/sessions/${SESSION}/output/ \
  demo/phase2/sessions/${SESSION}/output/
```

---

## What each step produces

### Razor `input/` (capture)

| Path | Content |
|------|---------|
| `session.json` | Schema 1.1 metadata |
| `rgb/left_rgb.png` | RGB frame |
| `depth/depth.npy` | Metric depth (m) |
| `calib/intrinsics.json`, `K.npy`, `extrinsics.json`, `robot_state.json` | Camera + robot at capture |
| `scene/table.json` | Table height |
| `segment/prompt.json` | Optional SAM point (needed for batch T2) |

### Titan `output/` (minimum for grasp)

| Path | Step |
|------|------|
| `output/status.json` | T7 — read first; `success: true` required |
| `output/inference/candidates.json` | T6 — grasp hypotheses (`mesh_frame: base_aligned`) |
| `output/register/T_base_mesh.json` | T5 |
| `output/mesh/object_base_aligned.glb` | T5 — collision / debug |
| `output/vis/T3_*.png` … `T6_*.png` | Review images |

### Razor execution

| Step | Script | Action |
|------|--------|--------|
| Load Titan output | `retarget.py` | `candidates.json` → `T_base_pinch` |
| IK filter | `pinch_ik.py` | Open-grip pre_grasp + grasp IK |
| Motion | `run_auto_grasp.py` → Phase 1 `executor.py` | OMPL approach, stall-close, lift |

---

## Prerequisites checklist

**Before first full run:**

- [ ] Razor: `start.yaml` tuned, SSH to Titan works
- [ ] Razor: `PIPELINE_TITAN_ROOT` points to Titan repo
- [ ] Titan: `segment_daemon` running (`python -m demo.pipeline.segment_daemon`)
- [ ] Titan: `demo/sessions/` exists, Razor SSH key authorized
- [ ] Titan: `python -m demo.pipeline --help` works in `bundlesdf`
- [ ] Both: test session rsync round-trip once

**Each experiment:**

- [ ] Object on table, lighting OK
- [ ] ZED stream running (if using default camera source)
- [ ] SSH tunnel ready for SAM2 when orchestrator shows `waiting_segment` (auto by default)

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Stuck after upload, no `daemon_state.json` | `segment_daemon` not running on Titan |
| SAM2 page won't load | Port 7860 in use by another app, or SSH key issue; try `--no-sam-popup` + manual tunnel |
| Titan T2 fails in `--ssh-pipeline` mode | Missing `input/segment/prompt.json` and no `output/segment/mask.png` |
| rsync permission denied | SSH key / Titan `authorized_keys` |
| `PIPELINE_TITAN_ROOT` error | Env not sourced; use `PIPELINE_TITAN_ROOT` not old `PIPELINE_UCB_ROOT` |
| All IK fail on Razor | Bad extrinsics, unreachable pose — inspect T6 vis + grasp preview |
| Popup does not block | Install `matplotlib`; fallback is OpenCV or Enter prompt |
| Titan timeout | Increase `PIPELINE_POLL_TIMEOUT_S`; check Titan `output/logs/process.log` |

---

## Related files

```text
demo/
├── PHASE2_PIPELINE.md              ← this file
├── phase2/
│   ├── run_server_client_pipeline.py   # Razor orchestrator
│   ├── run_auto_grasp.py               # Razor grasp execution
│   ├── capture_session.py              # Razor capture
│   ├── server_client_env.example       # env template
│   ├── SERVER_CLIENT_PLAN.md           # automation spec
│   └── sessions/                       # local session data (gitignored)
```

Titan side (separate repo):

```text
V2AP/                   # $TITAN_ROOT
├── demo/pipeline/                  # python -m demo.pipeline
├── demo/scripts/T1 … T7/
└── demo/sessions/                  # rsync target
```

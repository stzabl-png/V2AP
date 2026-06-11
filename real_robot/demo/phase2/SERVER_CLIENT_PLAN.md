# Titan ↔ Razor demo server–client pipeline

**Runbook:** [../PHASE2_PIPELINE.md](../PHASE2_PIPELINE.md)

**Audience:** Titan team (UCB_Project / GPU server) + Razor orchestrator.  
**Razor client:** `run_server_client_pipeline.py` + `server_client_env.example`.  
**Titan server:** `python -m demo.pipeline.process_razor_session` — see [UCB_Project `titan` branch `demo/`](https://github.com/stzabl-png/UCB_Project/tree/titan/demo).

**Related docs**

| Doc | Location | Purpose |
|-----|----------|---------|
| Full session schema (input/output) | [README.md](README.md) | Razor-side summary |
| What Razor needs from Titan | [TITAN_OUTPUT.md](TITAN_OUTPUT.md) | `status.json`, `candidates.json`, rsync minimum set |
| Titan authoritative spec | UCB `demo/README.md`, `demo/SERVER_CLIENT_PLAN.md` | T1–T7, GLB, PDM, `base_aligned` frame |

---

## 1. Goal

Automate the loop that is **manual rsync today**:

```text
[Human] place object
    → Razor capture (input/)
    → upload to Titan
    → Titan auto demo pipeline T1–T7 (output/)
    → download to Razor
    → Razor run_auto_grasp.py
```

**v1 transport:** SSH + rsync (no custom TCP server required).

---

## 2. Roles and connectivity

| Machine | Role | Network |
|---------|------|---------|
| **Razor** | Robot laptop; **orchestrator client**; initiates all connections | Often behind lab NAT — **does not need inbound SSH** |
| **Titan** | GPU server; **SSH server**; stores sessions; runs `python -m demo.pipeline.process_razor_session` | Stable hostname/IP or VPN |

**Connection direction:** Razor → Titan only.

---

## 3. SSH authentication (one-time setup)

### 3.1 On Razor (client)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/pipeline_razor_to_titan -C "razor-pipeline" -N ""
```

`~/.ssh/config` on Razor:

```text
Host titan-pipeline
    HostName <TITAN_HOSTNAME_OR_IP>
    User <TITAN_USER>          # e.g. vision
    IdentityFile ~/.ssh/pipeline_razor_to_titan
    IdentitiesOnly yes
```

Test:

```bash
ssh titan-pipeline 'echo OK && hostname && nvidia-smi -L | head -1'
```

### 3.2 On Titan (server)

Append Razor **public** key to `~/.ssh/authorized_keys` for `<TITAN_USER>`.

### 3.3 Environment variables (Razor orchestrator)

| Variable | Example | Purpose |
|----------|---------|---------|
| `PIPELINE_TITAN_SSH_HOST` | `titan-pipeline` or `vision@msc-vision` | SSH target |
| `PIPELINE_TITAN_ROOT` | `/path/to/V2AP` | Titan repo root (`$TITAN_ROOT`) |
| `PIPELINE_REMOTE_SESSIONS_SUBDIR` | `demo/sessions` | Under `$TITAN_ROOT` on Titan |
| `PIPELINE_RAZOR_REPO` | `/home/.../V2AP-demo` | This repo |
| `PIPELINE_TITAN_CONDA_ENV` | `bundlesdf` | Titan env for T4/T5/T6/T7 (T3 uses `sam3d-objects` internally) |
| `PIPELINE_TITAN_MODE` | `daemon` | `daemon` (default) or `ssh_pipeline` (legacy) |
| `PIPELINE_TITAN_SEGMENT_TUNNEL_PORT` | `7860` | Local port for SAM2 SSH tunnel during `waiting_segment` |
| `PIPELINE_TITAN_CMD` | see below | Legacy remote pipeline command |

Default remote command (see `server_client_env.example`):

```bash
export PIPELINE_TITAN_CMD='python -m demo.pipeline.process_razor_session --session-dir {session_dir} --device cuda'
```

`{session_dir}` is resolved as `demo/sessions/<session_id>` (relative to `PIPELINE_TITAN_ROOT`).

---

## 4. Directory layout (must mirror)

**Session ID:** `YYYYMMDD_HHMMSS_<object_slug>` (e.g. `20260602_192346_chips`).

| Side | Path |
|------|------|
| **Razor** | `$RAZOR_REPO/demo/phase2/sessions/<session_id>/` |
| **Titan** | `$TITAN_ROOT/demo/sessions/<session_id>/` |

Both:

```text
<session_id>/
├── input/          # Razor writes → Titan reads
└── output/         # Titan writes → Razor reads
```

Create once on Titan:

```bash
mkdir -p "$TITAN_ROOT/demo/sessions"
```

**Input contract:** [README.md § INPUT package](README.md#input-package-input--razor--titan) (`schema_version: "1.1"`).  
**Output contract:** [TITAN_OUTPUT.md](TITAN_OUTPUT.md) + UCB `demo/README.md`.

**Automated T2 (daemon mode, default):** Titan `segment_daemon` opens SAM2 Flask UI on `127.0.0.1:7860`; operator uses SSH tunnel from Razor. No `prompt.json` required.

**Batch T2 (legacy `--ssh-pipeline`):** requires **`input/segment/prompt.json`** OR pre-existing `output/segment/mask.png`. Without either, T2 fails (no GUI in batch mode).

---

## 5. Titan server (implemented on `titan` branch)

### 5.1 CLI entry point

```bash
cd "$TITAN_ROOT"
export FP_ROOT="$TITAN_ROOT/third_party/FoundationPose"
conda activate bundlesdf   # orchestrator also activates this env

python -m demo.pipeline.process_razor_session \
  --session-dir demo/sessions/<session_id> \
  [--skip-sam] [--skip-sam3d] [--skip-fp] \
  [--device cuda]
```

Equivalent: `python -m demo.pipeline --session-dir demo/sessions/<session_id>`

**Pipeline order:** T1 validate → T2 SAM → T3 SAM3D (`sam3d-objects`) → T4 scale → T5 FP + base align → T6 PDM → T7 status.

| Step | Output (minimum) |
|------|------------------|
| T2 | `output/segment/mask.png`, `prompt_used.json` |
| T3 | `output/mesh/object_raw.glb` |
| T4 | `output/mesh/object_scaled.glb`, `scale.json` |
| T5 | `object_base_aligned.glb`, `T_cam_mesh.json`, `T_base_mesh.json`, `mesh_frame_align.json` |
| T6 | `output/inference/candidates.json` (`mesh_frame: base_aligned`) |
| T7 | `output/status.json` (atomic, **last**) |

### 5.2 `output/status.json`

Razor **must** read this before grasp.

```json
{
  "schema_version": "1.1",
  "session_id": "20260602_192346_chips",
  "success": true,
  "pipeline_version": "demo.pipeline.process_razor_session 0.1.0",
  "steps": {
    "segment": "ok",
    "sam3d": "ok",
    "scale": "ok",
    "foundationpose": "ok",
    "grasp_pose": "ok"
  },
  "package": {
    "required_for_grasp": [
      "output/status.json",
      "output/inference/candidates.json",
      "output/register/T_base_mesh.json",
      "output/mesh/object_base_aligned.glb"
    ]
  }
}
```

Exit code **0** iff `success: true`. Logs: `output/logs/process.log`.

### 5.6 segment_daemon (default Razor → Titan flow)

Start once on Titan:

```bash
cd "$TITAN_ROOT" && conda activate bundlesdf
export FP_ROOT="$TITAN_ROOT/third_party/FoundationPose"
python -m demo.pipeline.segment_daemon
```

Razor orchestrator after rsync:

1. `ssh … cat > …/input/.upload_complete`
2. Poll `output/daemon_state.json` + `output/status.json`
3. When `state == waiting_segment`, orchestrator auto-starts `ssh -L` and opens browser (disable: `--no-sam-popup`)
4. After mask saved + Done, daemon runs T3–T7 and writes `success: true`

**Titan T6 (Razor → Titan contract):** read `input/session.json` → `pipeline.titan.max_candidates` (int, default **50**) → pass to `run_pdm_grasp.py` as `--n-samples`.

| File | `state` values |
|------|----------------|
| `output/status.json` | `waiting_segment`, `running`, done, `failed` |
| `output/daemon_state.json` | same + `segment_url` |

Env on Razor: `PIPELINE_TITAN_MODE=daemon`, `PIPELINE_TITAN_SEGMENT_TUNNEL_PORT=7860`.

### 5.3 Remote invocation (legacy ssh_pipeline)

```bash
ssh titan-pipeline "cd ${PIPELINE_TITAN_ROOT} && \
  source \$(conda info --base)/etc/profile.d/conda.sh && \
  conda activate ${PIPELINE_TITAN_CONDA_ENV} && \
  python -m demo.pipeline.process_razor_session \
    --session-dir demo/sessions/${SESSION_ID} \
    --device cuda"
```

Or use:

```bash
source demo/phase2/server_client_env.example
python demo/phase2/run_server_client_pipeline.py --object-name chips
```

### 5.4 Rsync commands (reference)

**Razor → Titan:**

```bash
rsync -avz --progress \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/input/" \
  "titan-pipeline:${PIPELINE_TITAN_ROOT}/demo/sessions/${SESSION}/input/"
```

**Titan → Razor:**

```bash
rsync -avz --progress \
  "titan-pipeline:${PIPELINE_TITAN_ROOT}/demo/sessions/${SESSION}/output/" \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/output/"
```

---

## 6. State machine

```text
CREATED          Razor: capture_session.py finished, input/ valid
UPLOADING        rsync input/ → Titan
MARKED           ssh mark input/.upload_complete
WAITING_SEGMENT  Titan daemon: SAM2 web UI (operator via SSH tunnel)
RUNNING          Titan: T3–T7 batch pipeline
DONE             status.json success=true
DOWNLOADING      rsync output/ ← Titan
READY_FOR_GRASP  Razor: run_auto_grasp.py
FAILED           any step; preserve logs + orchestrator_state.json
```

---

## 7. What Razor does (already implemented)

| Step | Razor script | Notes |
|------|--------------|-------|
| Capture | `capture_session.py` | `input/` pack, j3 spread pose |
| Orchestrate | `run_server_client_pipeline.py` | rsync + mark upload + poll + grasp |
| Grasp | `run_auto_grasp.py` | Requires `status.json` success |
| Retarget | open-grip IK | `candidates.json` → `T_base_pinch`; not Franka `position_panda_hand` |

---

## 8. Checklist

**Titan**

- [x] `demo/sessions/` rsync root + `.gitignore`
- [x] `python -m demo.pipeline.process_razor_session`
- [x] `status.json` + `candidates.json` + `object_base_aligned.glb`
- [ ] Razor SSH key in `authorized_keys`
- [ ] End-to-end test with fresh capture + `input/segment/prompt.json`

**Razor**

- [x] `run_server_client_pipeline.py` + env defaults aligned to `demo/sessions`
- [ ] `source demo/phase2/server_client_env.example` on lab laptop
- [ ] Full pipeline test once Titan SSH ready

---

## 9. Changelog

| Date | Notes |
|------|-------|
| 2026-06-03 | Align with UCB `titan` branch: `demo/sessions`, `demo.pipeline.process_razor_session`, PDM, GLB, `base_aligned` |
| 2026-06-03 | Initial plan: SSH client on Razor, server on Titan, rsync v1 |

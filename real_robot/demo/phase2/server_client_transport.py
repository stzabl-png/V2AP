"""rsync + ssh helpers for Phase 2 demo server–client pipeline."""

from __future__ import annotations

import json
import socket
import subprocess
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from demo.phase2.server_client_config import ServerClientPipelineConfig
from demo.phase2.session_markers import UPLOAD_COMPLETE_NAME

# Titan segment_daemon / status.json states that are not terminal failures.
_TITAN_IN_PROGRESS_STATES = frozenset(
    {
        None,
        "waiting_segment",
        "segment_done",
        "running",
        "queued",
    }
)


def _run(cmd: list[str]) -> None:
    logger.info(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def rsync_upload_input(cfg: ServerClientPipelineConfig, session_id: str) -> None:
    local = cfg.local_session_dir(session_id) / "input"
    if not local.is_dir():
        raise FileNotFoundError(f"Missing local input dir: {local}")
    remote = f"{cfg.titan_ssh_host}:{cfg.remote_session_dir(session_id)}/input/"
    _run(
        [
            "ssh",
            cfg.titan_ssh_host,
            f"mkdir -p '{cfg.remote_session_dir(session_id)}/input'",
        ]
    )
    _run(["rsync", "-avz", "--progress", f"{local}/", remote])
    logger.info(f"Uploaded input/ → {cfg.remote_session_dir(session_id)}/input/")


def rsync_download_output(cfg: ServerClientPipelineConfig, session_id: str) -> None:
    local = cfg.local_session_dir(session_id) / "output"
    local.mkdir(parents=True, exist_ok=True)
    remote = f"{cfg.titan_ssh_host}:{cfg.remote_session_dir(session_id)}/output/"
    _run(["rsync", "-avz", "--progress", remote, f"{local}/"])


def ssh_run_titan_pipeline(cfg: ServerClientPipelineConfig, session_id: str) -> int:
    """Run Titan server pipeline remotely (legacy); return exit code."""
    session_dir = cfg.remote_session_relpath(session_id)
    pipeline_cmd = cfg.titan_pipeline_cmd.format(session_dir=session_dir)
    remote_cmd = (
        f"cd {cfg.titan_root} && "
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {cfg.titan_conda_env} && "
        f"{pipeline_cmd}"
    )
    logger.info(f"Remote pipeline: {remote_cmd}")
    proc = subprocess.run(["ssh", cfg.titan_ssh_host, remote_cmd])
    return int(proc.returncode)


def ssh_mark_upload_complete(cfg: ServerClientPipelineConfig, session_id: str) -> None:
    """Write input/.upload_complete on Titan (no remote python required)."""
    payload = {
        "schema_version": "1.0",
        "marked_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": "razor",
    }
    body = json.dumps(payload, indent=2) + "\n"
    remote_input = f"{cfg.remote_session_dir(session_id)}/input"
    remote_path = f"{remote_input}/{UPLOAD_COMPLETE_NAME}"

    logger.info(f"Mark upload complete on Titan: {remote_path}")
    subprocess.run(
        ["ssh", cfg.titan_ssh_host, f"mkdir -p '{remote_input}'"],
        check=True,
    )
    proc = subprocess.run(
        ["ssh", cfg.titan_ssh_host, f"cat > '{remote_path}'"],
        input=body,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        raise RuntimeError(
            f"Failed to write {UPLOAD_COMPLETE_NAME} on Titan "
            f"(exit {proc.returncode}): {err}"
        )
    logger.info(f"Wrote {remote_path} on Titan")


def fetch_remote_json(
    cfg: ServerClientPipelineConfig, session_id: str, rel_path: str
) -> dict[str, Any] | None:
    remote_path = f"{cfg.remote_session_dir(session_id)}/{rel_path}"
    proc = subprocess.run(
        ["ssh", cfg.titan_ssh_host, f"cat '{remote_path}' 2>/dev/null || true"],
        capture_output=True,
        text=True,
    )
    text = proc.stdout.strip()
    if not text:
        return None
    return json.loads(text)


def fetch_remote_status(
    cfg: ServerClientPipelineConfig, session_id: str
) -> dict[str, Any] | None:
    return fetch_remote_json(cfg, session_id, "output/status.json")


def _titan_pipeline_state(
    status: dict[str, Any] | None, daemon: dict[str, Any] | None
) -> str | None:
    if daemon and daemon.get("state"):
        return str(daemon["state"])
    if status and status.get("state"):
        return str(status["state"])
    return None


def _localhost_port_open(port: int, *, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s):
            return True
    except OSError:
        return False


class _Sam2SshTunnel:
    """Background ssh -L for Titan segment_web.py (127.0.0.1:7860 on Titan)."""

    def __init__(self, cfg: ServerClientPipelineConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen[bytes] | None = None
        self._started_by_us = False

    @property
    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.cfg.titan_segment_tunnel_port}"

    def ensure_running(self) -> None:
        port = self.cfg.titan_segment_tunnel_port
        if _localhost_port_open(port):
            logger.info(f"SAM2 tunnel: reusing existing listener on port {port}")
            return

        cmd = [
            "ssh",
            "-N",
            "-L",
            f"{port}:127.0.0.1:{port}",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            self.cfg.titan_ssh_host,
        ]
        logger.info(f"Starting SAM2 SSH tunnel → {self.local_url}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._started_by_us = True

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"SSH tunnel exited with code {self._proc.returncode}. "
                    f"Check SSH to {self.cfg.titan_ssh_host!r}."
                )
            if _localhost_port_open(port):
                logger.info(f"SAM2 tunnel ready on port {port}")
                return
            time.sleep(0.25)
        raise RuntimeError(
            f"SSH tunnel did not open local port {port} within 20s. "
            "Is segment_web.py running on Titan?"
        )

    def close(self) -> None:
        if self._proc is None or not self._started_by_us:
            return
        if self._proc.poll() is None:
            logger.info("Closing SAM2 SSH tunnel")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def _open_sam2_popup(cfg: ServerClientPipelineConfig, tunnel: _Sam2SshTunnel) -> None:
    tunnel.ensure_running()
    logger.info(
        "\n=== Titan waiting for SAM2 segmentation ===\n"
        f"Opening {tunnel.local_url} — click object → Save mask → Done.\n"
        "Pipeline continues automatically after the mask is saved.\n"
    )
    webbrowser.open(tunnel.local_url)


def _print_sam2_manual_hint(cfg: ServerClientPipelineConfig) -> None:
    port = cfg.titan_segment_tunnel_port
    logger.info(
        "\n=== Titan waiting for SAM2 segmentation ===\n"
        "Auto popup disabled. Forward Titan's Flask UI manually:\n"
        f"  ssh -L {port}:127.0.0.1:{port} {cfg.titan_ssh_host}\n"
        f"Then open http://127.0.0.1:{port} — click object → Save mask → Done.\n"
    )


def wait_for_titan_success(
    cfg: ServerClientPipelineConfig,
    session_id: str,
    *,
    sam_popup: bool = True,
) -> dict[str, Any]:
    """Poll Titan until output/status.json reports success=true."""
    deadline = time.monotonic() + cfg.poll_timeout_s
    sam_popup_done = False
    tunnel = _Sam2SshTunnel(cfg) if sam_popup else None
    try:
        while time.monotonic() < deadline:
            status = fetch_remote_status(cfg, session_id)
            daemon = fetch_remote_json(cfg, session_id, "output/daemon_state.json")
            state = _titan_pipeline_state(status, daemon)

            if status is not None and status.get("success") is True:
                logger.info(f"Titan status.json: success=true ({session_id})")
                return status

            if state == "failed":
                msg = (daemon or {}).get("message") or status.get("errors")
                raise RuntimeError(f"Titan pipeline failed ({state}): {msg}")

            if status is not None and status.get("success") is False:
                if state not in _TITAN_IN_PROGRESS_STATES:
                    errors = status.get("errors", [])
                    raise RuntimeError(f"Titan pipeline failed: {errors}")

            if state == "waiting_segment" and not sam_popup_done:
                if tunnel is not None:
                    _open_sam2_popup(cfg, tunnel)
                else:
                    _print_sam2_manual_hint(cfg)
                sam_popup_done = True

            if cfg.titan_mode == "daemon" and status is None and daemon is None:
                hint = (
                    "Ensure Titan segment_daemon is running: "
                    f"cd {cfg.titan_root} && python -m demo.pipeline.segment_daemon"
                )
            else:
                hint = ""
            logger.info(
                f"Waiting for Titan (state={state!r}, {cfg.poll_interval_s:.0f}s)... "
                f"{hint}".rstrip()
            )
            time.sleep(cfg.poll_interval_s)
        raise TimeoutError(
            f"Titan did not write success status within {cfg.poll_timeout_s:.0f}s. "
            "Check segment_daemon and output/logs/ on Titan."
        )
    finally:
        if tunnel is not None:
            tunnel.close()


def write_orchestrator_state(session_dir: Path, state: dict[str, Any]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "orchestrator_state.json"
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

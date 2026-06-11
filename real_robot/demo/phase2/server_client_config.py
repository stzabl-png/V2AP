"""Environment config for Phase 2 demo server–client pipeline (Razor ↔ Titan)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from demo.phase2.constants import SESSIONS_DIR

_PHASE2_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PHASE2_DIR.parents[1]


@dataclass(frozen=True)
class ServerClientPipelineConfig:
    """Resolved paths for rsync + ssh orchestration."""

    titan_ssh_host: str
    titan_root: str
    remote_sessions_subdir: str
    razor_repo: Path
    sessions_dir: Path
    titan_conda_env: str
    titan_mode: str
    titan_segment_tunnel_port: int
    titan_auto_sam_popup: bool
    titan_pipeline_cmd: str
    poll_interval_s: float
    poll_timeout_s: float

    @property
    def remote_sessions_root(self) -> str:
        root = self.titan_root.rstrip("/")
        sub = self.remote_sessions_subdir.strip("/")
        return f"{root}/{sub}"

    def remote_session_dir(self, session_id: str) -> str:
        return f"{self.remote_sessions_root}/{session_id}"

    def local_session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def remote_session_relpath(self, session_id: str) -> str:
        sub = self.remote_sessions_subdir.strip("/")
        return f"{sub}/{session_id}"


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def load_server_client_pipeline_config() -> ServerClientPipelineConfig:
    titan_root = os.environ.get("PIPELINE_TITAN_ROOT", "").strip()
    if not titan_root:
        legacy = os.environ.get("PIPELINE_UCB_ROOT", "").strip()
        if legacy:
            titan_root = legacy
        else:
            raise RuntimeError(
                "Set PIPELINE_TITAN_ROOT to Titan project root "
                "(e.g. /path/to/V2AP)"
            )
    default_cmd = (
        "python -m demo.pipeline.process_razor_session "
        "--session-dir {session_dir} --device cuda"
    )
    titan_mode = os.environ.get("PIPELINE_TITAN_MODE", "daemon").strip().lower()
    if titan_mode not in ("daemon", "ssh_pipeline"):
        raise RuntimeError(
            "PIPELINE_TITAN_MODE must be 'daemon' or 'ssh_pipeline', "
            f"got {titan_mode!r}"
        )
    return ServerClientPipelineConfig(
        titan_ssh_host=os.environ.get("PIPELINE_TITAN_SSH_HOST", "titan-pipeline").strip(),
        titan_root=titan_root,
        remote_sessions_subdir=os.environ.get(
            "PIPELINE_REMOTE_SESSIONS_SUBDIR", "demo/sessions"
        ).strip(),
        razor_repo=Path(os.environ.get("PIPELINE_RAZOR_REPO", str(_PROJECT_ROOT))).resolve(),
        sessions_dir=Path(
            os.environ.get("PIPELINE_SESSIONS_DIR", str(SESSIONS_DIR))
        ).resolve(),
        titan_conda_env=os.environ.get("PIPELINE_TITAN_CONDA_ENV", "bundlesdf").strip(),
        titan_mode=titan_mode,
        titan_segment_tunnel_port=int(
            os.environ.get("PIPELINE_TITAN_SEGMENT_TUNNEL_PORT", "7860")
        ),
        titan_auto_sam_popup=_env_bool("PIPELINE_TITAN_AUTO_SAM_POPUP", True),
        titan_pipeline_cmd=os.environ.get("PIPELINE_TITAN_CMD", default_cmd).strip(),
        poll_interval_s=float(os.environ.get("PIPELINE_POLL_INTERVAL_S", "15")),
        poll_timeout_s=float(os.environ.get("PIPELINE_POLL_TIMEOUT_S", "3600")),
    )

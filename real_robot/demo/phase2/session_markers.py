"""Upload markers for Titan segment_daemon (must match V2AP demo/pipeline/session_markers.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UPLOAD_COMPLETE_NAME = ".upload_complete"
UPLOAD_PROCESSED_NAME = ".upload_processed"


def upload_complete_path(session_root: Path) -> Path:
    return session_root / "input" / UPLOAD_COMPLETE_NAME


def upload_processed_path(session_root: Path) -> Path:
    return session_root / "input" / UPLOAD_PROCESSED_NAME


def write_upload_complete(
    session_root: Path,
    *,
    source: str = "razor",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Signal Titan segment_daemon that rsync input/ is finished."""
    path = upload_complete_path(session_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "marked_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": source,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

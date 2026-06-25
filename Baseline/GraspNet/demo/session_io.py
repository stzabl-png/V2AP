#!/usr/bin/env python3
"""
graspnet_demo/session_io.py
===========================
Session path management for GraspNet-Demo.

Mirrors V2AP-demo demo/phase2/session_io.py but adapted for our
local graspnet_demo/sessions/ directory layout.
"""
from __future__ import annotations

import json
from pathlib import Path

# Default sessions dir next to this file
_DEMO_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = _DEMO_DIR / "sessions"


def session_dir_for_id(session_id: str, sessions_dir: Path | None = None) -> Path:
    """Return the session directory for a given session ID.

    Searches in order:
    1. Absolute path if session_id looks like one
    2. graspnet_demo/sessions/<session_id>
    3. /media/lyh/KINGSTON/<session_id>  (USB fallback)
    """
    # Absolute path given directly
    if Path(session_id).is_absolute() and Path(session_id).exists():
        return Path(session_id)

    base = sessions_dir or SESSIONS_DIR

    # Local sessions dir
    local = base / session_id
    if local.exists():
        return local

    # USB fallback
    usb = Path("/media/lyh/KINGSTON") / session_id
    if usb.exists():
        return usb

    raise FileNotFoundError(
        f"Session '{session_id}' not found in:\n"
        f"  {local}\n"
        f"  {usb}\n"
        f"Use --session-dir <absolute_path> to specify directly."
    )


def output_dir(session_dir: Path) -> Path:
    return session_dir / "output"


def inference_dir(session_dir: Path) -> Path:
    return session_dir / "output" / "inference"


def candidates_path(session_dir: Path) -> Path:
    return inference_dir(session_dir) / "candidates.json"


def status_path(session_dir: Path) -> Path:
    return output_dir(session_dir) / "status.json"


def load_candidates(session_dir: Path) -> dict:
    p = candidates_path(session_dir)
    if not p.exists():
        raise FileNotFoundError(f"candidates.json not found: {p}")
    with open(p) as f:
        return json.load(f)


def load_status(session_dir: Path) -> dict:
    p = status_path(session_dir)
    if not p.exists():
        raise FileNotFoundError(f"status.json not found: {p}")
    with open(p) as f:
        return json.load(f)

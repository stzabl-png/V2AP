"""Titan-side hints embedded in Razor input/session.json (read by Affordance2Grasp T6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from demo.phase2.constants import DEFAULT_TITAN_MAX_CANDIDATES

TITAN_MAX_CANDIDATES_KEY = ("pipeline", "titan", "max_candidates")


def read_titan_max_candidates(session_json: dict[str, Any]) -> int | None:
    """Return ``pipeline.titan.max_candidates`` if present."""
    pipeline = session_json.get("pipeline")
    if not isinstance(pipeline, dict):
        return None
    titan = pipeline.get("titan")
    if not isinstance(titan, dict):
        return None
    val = titan.get("max_candidates")
    if val is None:
        return None
    return int(val)


def patch_input_titan_options(
    input_dir: Path,
    *,
    max_candidates: int = DEFAULT_TITAN_MAX_CANDIDATES,
) -> Path:
    """
    Set ``pipeline.titan.max_candidates`` on an existing ``input/session.json``.

    Titan T6 should read this and pass ``--n-samples`` to PDM (default 50 if absent).
    """
    input_dir = Path(input_dir)
    session_path = input_dir / "session.json"
    if not session_path.is_file():
        raise FileNotFoundError(f"Missing {session_path}")

    n = int(max_candidates)
    if n < 1:
        raise ValueError(f"max_candidates must be >= 1, got {n}")

    data = json.loads(session_path.read_text(encoding="utf-8"))
    pipeline = data.setdefault("pipeline", {})
    if not isinstance(pipeline, dict):
        pipeline = {}
        data["pipeline"] = pipeline
    titan = pipeline.setdefault("titan", {})
    if not isinstance(titan, dict):
        titan = {}
        pipeline["titan"] = titan
    titan["max_candidates"] = n

    session_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return session_path

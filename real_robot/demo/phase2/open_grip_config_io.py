"""Load open-grip arm retarget parameters (Titan pinch → open-hand IK)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from demo.phase2.constants import CALIB_DIR

DEFAULT_OPEN_GRIP_YAML = CALIB_DIR / "open_grip.yaml"


@dataclass
class OpenGripConfig:
    """Open-grip pinch alignment parameters for ``pinch_ik.py``."""

    open_pinch_forward_offset_m: float = 0.015
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_pinch_forward_offset_m": float(self.open_pinch_forward_offset_m),
            "notes": self.notes,
        }


def load_open_grip_config(path: str | Path | None = None) -> OpenGripConfig:
    path = Path(path or DEFAULT_OPEN_GRIP_YAML)
    if not path.is_file():
        return OpenGripConfig()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"open_grip config must be a YAML mapping: {path}")
    return OpenGripConfig(
        open_pinch_forward_offset_m=float(raw.get("open_pinch_forward_offset_m", 0.015)),
        notes=str(raw.get("notes", "")),
    )


def save_open_grip_config(
    config: OpenGripConfig, path: str | Path | None = None
) -> Path:
    path = Path(path or DEFAULT_OPEN_GRIP_YAML)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, default_flow_style=False)
    return path

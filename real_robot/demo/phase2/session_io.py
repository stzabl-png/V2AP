"""Load Phase 2 session input/ packages from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SessionInput:
    input_dir: Path
    session: dict[str, Any]
    rgb: np.ndarray  # (H,W,3) uint8 RGB
    depth: np.ndarray  # (H,W) float32 meters
    K: np.ndarray  # (3,3)
    T_base_cam: np.ndarray  # (4,4)
    robot_state: dict[str, Any]
    table: dict[str, Any]
    intrinsics: dict[str, Any]
    extrinsics: dict[str, Any]
    segment_prompt: dict[str, Any] | None = None


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_session_input(input_dir: Path) -> SessionInput:
    """Load a captured input/ folder (rgb, depth, calib, scene)."""
    input_dir = Path(input_dir)
    session = _load_json(input_dir / "session.json")
    intrinsics = _load_json(input_dir / "calib" / "intrinsics.json")
    extrinsics = _load_json(input_dir / "calib" / "extrinsics.json")
    robot_state = _load_json(input_dir / "calib" / "robot_state.json")
    table = _load_json(input_dir / "scene" / "table.json")

    segment_prompt = None
    prompt_path = input_dir / "segment" / "prompt.json"
    if prompt_path.is_file():
        segment_prompt = _load_json(prompt_path)

    try:
        import cv2  # noqa: WPS433

        bgr = cv2.imread(str(input_dir / "rgb" / "left_rgb.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError("rgb/left_rgb.png unreadable")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image  # noqa: WPS433

        rgb = np.array(Image.open(input_dir / "rgb" / "left_rgb.png").convert("RGB"))

    depth = np.load(input_dir / "depth" / "depth.npy").astype(np.float32)
    K = np.asarray(intrinsics["K"], dtype=np.float64)
    T_base_cam = np.asarray(extrinsics["T_base_cam"], dtype=np.float64)

    return SessionInput(
        input_dir=input_dir,
        session=session,
        rgb=rgb,
        depth=depth,
        K=K,
        T_base_cam=T_base_cam,
        robot_state=robot_state,
        table=table,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        segment_prompt=segment_prompt,
    )

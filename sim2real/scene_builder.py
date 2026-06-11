"""
sim2real/scene_builder.py

Dual-robot scene for retarget validation.
Left  (original position) : Franka + Gripper
Right (+X offset)          : Dexmate Vega + Sharpa Wave

Both tables carry the same object, both robots target the same grasp candidate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

SIM2REAL_DIR = Path(__file__).resolve().parent
PROJ_DIR = SIM2REAL_DIR.parent
SIM_DIR = PROJ_DIR / "sim"

# ── path helpers ──────────────────────────────────────────────────────────────
for _p in [str(SIM_DIR), str(PROJ_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── layout constants ──────────────────────────────────────────────────────────
# Table A – Franka (original scene, identical to sim/evaluation/scene_builder.py)
FRANKA_ROBOT_POSITION    = [0.2, -0.05, 0.8]
FRANKA_ROBOT_ORIENTATION = [0.0, 0.0, 90.0]  # euler-xyz deg
TABLE_A_POSITION         = [0.0, 1.0, 0.75]
TABLE_ORIENTATION        = [0.0, 0.0, 0.0]
TABLE_SCALE              = [2.0, 2.0, 0.1]
TABLE_TOP_Z              = 0.80
OBJECT_BASE_POSITION     = [0.0, 0.55, TABLE_TOP_Z]

# Table B – Dexmate (shifted +X)
TABLE_X_OFFSET = 2.0          # metres between the two setups
TABLE_B_POSITION = [TABLE_A_POSITION[0] + TABLE_X_OFFSET,
                    TABLE_A_POSITION[1],
                    TABLE_A_POSITION[2]]
DEXMATE_ROBOT_POSITION    = [FRANKA_ROBOT_POSITION[0] + TABLE_X_OFFSET,
                              FRANKA_ROBOT_POSITION[1],
                              0.0]                          # ground level (z=0)
DEXMATE_ROBOT_ORIENTATION = [0.0, 0.0, 90.0]               # 90° CCW around Z (same as Franka)

# Asset paths
_ASSETS = SIM2REAL_DIR / "assets_dexmate"
_SHARPA = SIM2REAL_DIR / "assets_sharpa"
# Dexmate: USD not in OSS repo; use URDF (Isaac Sim can import it at runtime)
DEXMATE_URDF_PATH = str(_ASSETS / "vega_1" / "vega_1.urdf")
DEXMATE_USD_PATH  = str(_ASSETS / "vega_1" / "vega_1.usd")   # only if manually placed
# Sharpa: USDA available directly from cloned repo
SHARPA_USD_PATH   = str(_SHARPA / "right_sharpa_wave" / "right_sharpa_wave_with_flange.usda")
SHARPA_URDF_PATH  = str(_SHARPA / "right_sharpa_wave_with_flange.urdf")

# Override file (reuse from sim/)
OVERRIDE_FILE = SIM_DIR / "object_rotation_overrides.json"
try:
    with open(OVERRIDE_FILE, encoding="utf-8") as _f:
        _raw = json.load(_f)
    OBJECT_ROTATION_OVERRIDES = {k: v for k, v in _raw.items() if not str(k).startswith("_")}
except Exception:
    OBJECT_ROTATION_OVERRIDES = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_obj_usd_path(obj_id: str) -> str | None:
    """Search the same locations as the original scene_builder."""
    obj_usd_root = PROJ_DIR / "output" / "obj_usd"
    datasets = ["oakink", "ycb", "arctic", "dexycb", "egocentric", "ho3d_v3", "unseen", "egodex"]
    search = (
        [obj_usd_root / ds / f"{obj_id}.usd" for ds in datasets]
        + [SIM_DIR / "assets" / f"{obj_id}.usd"]
    )
    return next((str(p) for p in search if p.exists()), None)


def _euler_xyz_deg_to_wxyz(euler_deg: list[float]) -> list[float]:
    q = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def resolve_object_placement(
    obj_id: str,
    object_scale: float,
    sim_z_yaw_deg: float = 0.0,
    x_offset: float = 0.0,
) -> dict[str, Any]:
    """Placement dict for one object, with optional X shift for Table B."""
    import os
    override = OBJECT_ROTATION_OVERRIDES.get(obj_id)
    obj_orientation = [0.0, 0.0, 0.0]
    usd_path = find_obj_usd_path(obj_id)
    if usd_path is None:
        raise FileNotFoundError(f"USD not found for {obj_id}")

    meta_path = usd_path.replace(".usd", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        z_offset = float(meta.get("z_offset_m", 0.075 * object_scale))
    elif isinstance(override, dict) and "z_offset" in override:
        z_offset = float(override["z_offset"])
    else:
        z_offset = 0.075 * object_scale

    if isinstance(override, dict) and "rotation" in override:
        obj_orientation = list(override["rotation"])

    if abs(sim_z_yaw_deg) > 1e-9:
        obj_orientation[2] = float(obj_orientation[2]) + sim_z_yaw_deg

    pos = list(OBJECT_BASE_POSITION)
    pos[0] += x_offset
    pos[2] += z_offset
    return {"pos": pos, "ori": obj_orientation, "z_offset": z_offset, "usd_path": usd_path}

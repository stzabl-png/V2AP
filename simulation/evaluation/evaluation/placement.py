"""Object XY placement helpers for evaluation (Sim spawn only)."""

from __future__ import annotations

import argparse

import numpy as np

from evaluation.randomness import DEFAULT_EVAL_SEED, mix_eval_seed

DEFAULT_OBJ_XY_JITTER_M = 0.05


def add_random_obj_xy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--random-obj-xy",
        action="store_true",
        help="Randomize object spawn XY around the default table position (Z unchanged).",
    )
    parser.add_argument(
        "--obj-xy-jitter-m",
        type=float,
        default=DEFAULT_OBJ_XY_JITTER_M,
        help="Uniform half-range (m) for --random-obj-xy: dx,dy ~ U[-jitter,+jitter].",
    )


def resolve_obj_xy_offset(
    *,
    random_obj_xy: bool,
    obj_xy_jitter_m: float,
    obj_id: str,
    trial: int,
    sim_z_yaw_deg: float,
    obj_xy_offset: list[float] | tuple[float, float] | None = None,
    eval_seed: int = DEFAULT_EVAL_SEED,
) -> tuple[float, float]:
    """Return (dx, dy) added to OBJECT_POSITION x/y in world frame."""
    if obj_xy_offset is not None:
        arr = np.asarray(obj_xy_offset, dtype=np.float64).reshape(2)
        return float(arr[0]), float(arr[1])
    jitter = float(obj_xy_jitter_m)
    if not random_obj_xy or jitter <= 0.0:
        return 0.0, 0.0
    mixed = mix_eval_seed(eval_seed, "obj_xy", obj_id, int(trial), int(round(float(sim_z_yaw_deg))) % 360)
    rng = np.random.default_rng(mixed)
    return float(rng.uniform(-jitter, jitter)), float(rng.uniform(-jitter, jitter))

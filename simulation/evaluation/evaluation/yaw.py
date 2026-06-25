"""Sim z-yaw resolution for evaluation episodes."""

from __future__ import annotations

from typing import Sequence

from evaluation.randomness import DEFAULT_EVAL_SEED, mix_eval_seed


def parse_yaw_pool(text: str | None) -> list[float]:
    if not text or not str(text).strip():
        return [0.0, 90.0, 180.0, 270.0]
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(float(part) % 360.0)
    return out or [0.0]


def np_random_for_draw(obj_id: str, trial: int, *, eval_seed: int = DEFAULT_EVAL_SEED):
    import numpy as np

    return np.random.default_rng(mix_eval_seed(eval_seed, "yaw_draw", obj_id, int(trial)))


def resolve_z_yaw_deg(
    *,
    trial: int,
    obj_id: str,
    z_yaw_deg: float | None = None,
    z_yaw_grid: Sequence[float] | None = None,
    z_yaw_random_pool: Sequence[float] | None = None,
    z_yaw_random: bool = False,
    eval_seed: int = DEFAULT_EVAL_SEED,
) -> float:
    """Pick sim / PDM conditioning yaw for one episode."""
    if z_yaw_deg is not None:
        return float(z_yaw_deg) % 360.0
    if z_yaw_grid:
        grid = [float(y) % 360.0 for y in z_yaw_grid]
        return grid[int(trial) % len(grid)]
    if z_yaw_random:
        pool = list(z_yaw_random_pool or [0.0, 90.0, 180.0, 270.0])
        rng = np_random_for_draw(obj_id, trial, eval_seed=eval_seed)
        return float(pool[int(rng.integers(0, len(pool)))])
    return 0.0

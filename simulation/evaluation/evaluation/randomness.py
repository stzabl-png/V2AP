"""Evaluation RNG helpers (unified --seed with optional per-trial offsets)."""

from __future__ import annotations

import argparse
import hashlib
import secrets

DEFAULT_EVAL_SEED = 42


def add_eval_seed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Master RNG seed for eval randomness (policy sample, random XY, yaw draw, etc.).",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="Override seed for --selection sample only; default uses --seed.",
    )


def _tag_to_int(tag: str) -> int:
    """Process-stable string tag → 32-bit int (do not use built-in hash())."""

    return int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16)


def mix_eval_seed(base: int, *parts) -> int:
    """Deterministically mix base seed with string/int tags into a 32-bit seed."""

    h = int(base) & 0xFFFFFFFF
    for part in parts:
        if isinstance(part, str):
            h ^= _tag_to_int(part) & 0xFFFFFFFF
        else:
            h ^= (int(part) * 2654435761) & 0xFFFFFFFF
    return int(h % (2**32))


def fresh_rng():
    import numpy as np

    return np.random.default_rng(secrets.randbits(128))


def fresh_seed() -> int:
    return secrets.randbelow(2**31 - 1)


def resolve_policy_seed(
    *,
    eval_seed: int = DEFAULT_EVAL_SEED,
    policy_seed: int | None = None,
    trial: int | None = None,
) -> int:
    """Return an integer seed for policy HDF5 selection."""

    base = int(policy_seed if policy_seed is not None else eval_seed)
    if trial is not None:
        return base + int(trial) * 10007
    return base


def record_trials_rng(*, eval_seed: int, obj_id: str):
    import numpy as np

    return np.random.default_rng(mix_eval_seed(eval_seed, "record_trials", obj_id))


def shuffle_objects_rng(*, eval_seed: int):
    import numpy as np

    return np.random.default_rng(mix_eval_seed(eval_seed, "shuffle_objects"))

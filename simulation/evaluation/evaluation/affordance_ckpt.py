"""Resolve affordance v6 checkpoint paths for evaluation / candidate generation."""

from __future__ import annotations

from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

DEFAULT_AFFORDANCE_CHECKPOINT = (
    PROJ / "output" / "affordance_no_rot_executed" / "min20" / "checkpoints_v6" / "best_v6_model.pth"
)
HP_AFFORDANCE_CHECKPOINT = (
    PROJ / "output" / "affordance_hp_v6" / "min20" / "checkpoints_v6" / "best_v6_model.pth"
)


def resolve_affordance_checkpoint(
    *,
    hp_affordance: bool = False,
    affordance_checkpoint: str | Path | None = None,
) -> Path:
    """Explicit --affordance-checkpoint wins; else --hp-affordance selects HP ckpt; else default."""
    if affordance_checkpoint is not None:
        path = Path(affordance_checkpoint).expanduser().resolve()
    elif hp_affordance:
        path = HP_AFFORDANCE_CHECKPOINT.resolve()
    else:
        path = DEFAULT_AFFORDANCE_CHECKPOINT.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Affordance checkpoint not found: {path}")
    return path


def add_affordance_checkpoint_args(parser) -> None:
    parser.add_argument(
        "--hp-affordance",
        action="store_true",
        help=(
            "Use human-prior affordance v6 checkpoint "
            f"({HP_AFFORDANCE_CHECKPOINT.relative_to(PROJ)}). "
            "Default (off): executed-soft affordance checkpoint."
        ),
    )
    parser.add_argument(
        "--affordance-checkpoint",
        type=Path,
        default=None,
        help=(
            "Override affordance v6 weights. If omitted: default executed-soft ckpt, "
            "or HP ckpt when --hp-affordance is set."
        ),
    )

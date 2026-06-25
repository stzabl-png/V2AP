#!/usr/bin/env python3
"""Build evaluation object CSV from training success counts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
if str(PROJ / "tools") not in sys.path:
    sys.path.insert(0, str(PROJ / "tools"))

from grasp_pool_common import (  # noqa: E402
    scan_merged_trusted_objects,
    scan_success_round_ge3,
)

DEFAULT_OUTDIR = PROJ / "output" / "grasp_collect_no_rot"
DEFAULT_OUTPUT = PROJ / "evaluation" / "configs" / "eval_objects_merged_success_ge30.csv"


def build_list(
    *,
    source: str,
    outdir: Path,
    merged_dir: Path,
    min_success: int,
    min_round: int,
) -> tuple[list[dict[str, str]], str]:
    if source == "robot_gt":
        counts = scan_success_round_ge3(str(outdir), min_round=int(min_round))
        note_tag = f"round_ge{min_round}_success_ge{min_success}"
        criterion = f"round>={min_round} robot_gt success_count>={min_success}"
    elif source == "merged":
        counts = scan_merged_trusted_objects(str(merged_dir))
        note_tag = f"merged_trusted_ge{min_success}"
        criterion = f"merged trusted success_count>={min_success} ({merged_dir})"
    else:
        raise ValueError(f"unknown source: {source}")

    rows = [
        {
            "obj_id": obj_id,
            "enabled": "1",
            "success_count": str(int(n)),
            "notes": note_tag,
        }
        for obj_id, n in counts.items()
        if int(n) >= int(min_success)
    ]
    rows.sort(key=lambda r: (-int(r["success_count"]), r["obj_id"]))
    return rows, criterion


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build eval object list from training success counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=("robot_gt", "merged"),
        default="robot_gt",
        help="robot_gt: sum successful_grasps in robot_gt/round_R (R>=min-round). "
        "merged: count gripper_tips_trusted successes in merged HDF5.",
    )
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Merged GT directory (default: <outdir>/merged). Used with --source merged.",
    )
    p.add_argument("--min-success", type=int, default=20)
    p.add_argument(
        "--min-round",
        type=int,
        default=3,
        help="robot_gt only: count successes from round_R with R >= this",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    outdir = args.outdir.expanduser().resolve()
    merged_dir = (
        args.merged_dir.expanduser().resolve()
        if args.merged_dir is not None
        else (outdir / "merged")
    )

    rows, criterion = build_list(
        source=str(args.source),
        outdir=outdir,
        merged_dir=merged_dir,
        min_success=int(args.min_success),
        min_round=int(args.min_round),
    )
    if not rows:
        raise SystemExit(f"No objects matched ({criterion})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["obj_id", "enabled", "success_count", "notes"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} objects -> {args.output}")
    print(f"  source: {args.source}")
    print(f"  criterion: {criterion}")
    print(f"  success_count range: {rows[-1]['success_count']} .. {rows[0]['success_count']}")


if __name__ == "__main__":
    main()

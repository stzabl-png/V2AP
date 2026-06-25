#!/usr/bin/env python3
"""Summarize eval results by setup and training-data GE tier (micro success rate)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_GE_DIR = PROJ / "evaluation" / "configs"
DEFAULT_FG_DIR = PROJ / "output" / "evaluation" / "flying_gripper"
DEFAULT_ROUND_ROOT = PROJ / "output" / "round_eval"
DEFAULT_OUTPUT = PROJ / "output" / "evaluation" / "setup_ge_summary.csv"

GE_THRESHOLDS = (30, 35, 40, 45, 50)
SETUP_TO_LABEL = {
    "1a": "hp",
    "1b": "hp",
    "2a": "rp",
    "2b": "rp",
}
ROUND_SETUPS = ("1a", "1b", "2a", "2b")


def load_ge_ids(ge_dir: Path, threshold: int) -> set[str]:
    path = ge_dir / f"eval_objects_merged_success_ge{threshold}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("enabled", "1")).strip() in ("0", "false", "False"):
                continue
            oid = str(row.get("obj_id", "")).strip()
            if oid:
                ids.add(oid)
    return ids


def micro_rate(entries: dict[str, tuple[int, int]]) -> tuple[float, int]:
    """Return (success_rate, n_objects) from obj_id -> (success, total)."""
    if not entries:
        return float("nan"), 0
    n_succ = sum(v[0] for v in entries.values())
    n_total = sum(v[1] for v in entries.values())
    if n_total <= 0:
        return float("nan"), len(entries)
    return n_succ / n_total, len(entries)


def filter_by_ids(
    entries: dict[str, tuple[int, int]], allowed: set[str]
) -> dict[str, tuple[int, int]]:
    return {k: v for k, v in entries.items() if k in allowed}


def load_flying_gripper(path: Path) -> dict[str, tuple[int, int]]:
    """FG JSON: success_rate = num_valid_poses / num_input_poses (typically 500)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int]] = {}
    for row in data.get("objects", []):
        if not row.get("processed", True):
            continue
        oid = str(row["obj_id"])
        n_total = int(row.get("num_input_poses", 500))
        rate = float(row.get("success_rate", 0.0))
        n_succ = int(row.get("num_valid_poses", round(rate * n_total)))
        out[oid] = (n_succ, n_total)
    return out


def load_eval_summary(path: Path) -> dict[str, tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int]] = {}
    for oid, stats in (data.get("by_object") or {}).items():
        out[str(oid)] = (int(stats.get("success", 0)), int(stats.get("total", 0)))
    return out


def merge_entries(parts: list[dict[str, tuple[int, int]]]) -> dict[str, tuple[int, int]]:
    merged: dict[str, tuple[int, int]] = {}
    for part in parts:
        for oid, (succ, total) in part.items():
            ps, pt = merged.get(oid, (0, 0))
            merged[oid] = (ps + succ, pt + total)
    return merged


def discover_round_summaries(root: Path, setup_id: str) -> list[Path]:
    pat = re.compile(rf"^(\d+)_{re.escape(setup_id)}$")
    found: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if not m:
            continue
        summary = child / "eval_summary.json"
        if summary.is_file():
            found.append((int(m.group(1)), summary))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


def rows_for_ge_split(
    *,
    setup: str,
    entries: dict[str, tuple[int, int]],
    ge_dir: Path,
    thresholds: tuple[int, ...],
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for t in thresholds:
        allowed = load_ge_ids(ge_dir, t)
        subset = filter_by_ids(entries, allowed)
        rate, n_obj = micro_rate(subset)
        rows.append(
            {
                "setup": setup,
                "dataset": f"ge{t}",
                "success_rate": rate,
                "n_objects": n_obj,
            }
        )
    return rows


def rows_for_unseen_split(
    *, setup: str, entries: dict[str, tuple[int, int]]
) -> list[dict[str, str | float | int]]:
    rate, n_obj = micro_rate(entries)
    return [
        {
            "setup": setup,
            "dataset": "unseen",
            "success_rate": rate,
            "n_objects": n_obj,
        }
    ]


def collect_flying_gripper(fg_dir: Path, ge_dir: Path, thresholds: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    ge_path = fg_dir / "eval_objects_ge30_success_rates.json"
    unseen_path = fg_dir / "eval_unseen_objects_success_rates.json"
    if ge_path.is_file():
        rows.extend(
            rows_for_ge_split(
                setup="fg",
                entries=load_flying_gripper(ge_path),
                ge_dir=ge_dir,
                thresholds=thresholds,
            )
        )
    if unseen_path.is_file():
        rows.extend(rows_for_unseen_split(setup="fg", entries=load_flying_gripper(unseen_path)))
    return rows


def collect_round_eval(
    root: Path, ge_dir: Path, thresholds: tuple[int, ...], setups: tuple[str, ...]
) -> list[dict]:
    rows: list[dict] = []
    for setup_id in setups:
        label = SETUP_TO_LABEL[setup_id]
        summaries = discover_round_summaries(root, setup_id)
        if not summaries:
            continue
        merged = merge_entries([load_eval_summary(p) for p in summaries])
        if setup_id in ("1a", "2a"):
            rows.extend(
                rows_for_ge_split(
                    setup=label,
                    entries=merged,
                    ge_dir=ge_dir,
                    thresholds=thresholds,
                )
            )
        else:
            rows.extend(rows_for_unseen_split(setup=label, entries=merged))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setup", "dataset", "success_rate", "n_objects"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "setup": row["setup"],
                    "dataset": row["dataset"],
                    "success_rate": f"{float(row['success_rate']):.6f}",
                    "n_objects": int(row["n_objects"]),
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize eval by setup and GE tier")
    p.add_argument("--ge-dir", type=Path, default=DEFAULT_GE_DIR)
    p.add_argument("--flying-gripper-dir", type=Path, default=DEFAULT_FG_DIR)
    p.add_argument("--round-eval-root", type=Path, default=DEFAULT_ROUND_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in GE_THRESHOLDS),
        help="Comma-separated GE tiers, e.g. 30,35,40,45,50",
    )
    args = p.parse_args()

    thresholds = tuple(int(x.strip()) for x in str(args.thresholds).split(",") if x.strip())
    ge_dir = args.ge_dir.expanduser().resolve()

    rows: list[dict] = []
    rows.extend(collect_flying_gripper(args.flying_gripper_dir.expanduser().resolve(), ge_dir, thresholds))
    rows.extend(
        collect_round_eval(
            args.round_eval_root.expanduser().resolve(),
            ge_dir,
            thresholds,
            ROUND_SETUPS,
        )
    )

    # Stable order: fg, hp, rp; ge30..ge50 then unseen
    dataset_order = {f"ge{t}": i for i, t in enumerate(thresholds)}
    dataset_order["unseen"] = len(thresholds)
    setup_order = {"fg": 0, "hp": 1, "rp": 2}
    rows.sort(key=lambda r: (setup_order.get(str(r["setup"]), 9), dataset_order.get(str(r["dataset"]), 99)))

    out = args.output.expanduser().resolve()
    write_csv(out, rows)
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()

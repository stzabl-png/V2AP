#!/usr/bin/env python3
"""Run multi-round evaluation ablations (1a–4b) via eval_pool and aggregate results to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJ / "output" / "round_eval"
DEFAULT_SUMMARY_CSV = "round_eval_summary.csv"
DEFAULT_ROUNDS = 10
DEFAULT_TRIALS = 5
SETUP_ORDER = ("1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b")

GE30_CSV = PROJ / "evaluation" / "configs" / "eval_objects_merged_success_ge30.csv"
UNSEEN_CSV = PROJ / "evaluation" / "configs" / "eval_unseen_all.csv"

SUMMARY_FIELDS = [
    "round",
    "setup_id",
    "status",
    "result_dir",
    "obj_list",
    "hp_affordance",
    "filtering",
    "trials_per_obj_yaw",
    "n_objects",
    "n_episodes",
    "n_success",
    "success_rate",
    "n_obj_success_rate_1.0",
    "n_obj_success_rate_ge_0.8",
    "n_obj_success_rate_ge_0.6",
    "n_obj_success_rate_ge_0.4",
    "n_obj_success_rate_eq_0",
    "failure_stages_json",
    "started_at",
    "finished_at",
    "duration_s",
    "returncode",
    "notes",
]


@dataclass(frozen=True)
class SetupSpec:
    setup_id: str
    obj_list: Path
    hp_affordance: bool
    no_filtering: bool
    description: str


SETUPS: dict[str, SetupSpec] = {
    "1a": SetupSpec(
        "1a",
        GE30_CSV,
        hp_affordance=True,
        no_filtering=True,
        description="HP affordance, no filter, GE30",
    ),
    "1b": SetupSpec(
        "1b",
        UNSEEN_CSV,
        hp_affordance=True,
        no_filtering=True,
        description="HP affordance, no filter, unseen",
    ),
    "2a": SetupSpec(
        "2a",
        GE30_CSV,
        hp_affordance=False,
        no_filtering=False,
        description="default affordance, with filter, GE30",
    ),
    "2b": SetupSpec(
        "2b",
        UNSEEN_CSV,
        hp_affordance=False,
        no_filtering=False,
        description="default affordance, with filter, unseen",
    ),
    "3a": SetupSpec(
        "3a",
        GE30_CSV,
        hp_affordance=False,
        no_filtering=True,
        description="default affordance, no filter, GE30",
    ),
    "3b": SetupSpec(
        "3b",
        UNSEEN_CSV,
        hp_affordance=False,
        no_filtering=True,
        description="default affordance, no filter, unseen",
    ),
    "4a": SetupSpec(
        "4a",
        GE30_CSV,
        hp_affordance=True,
        no_filtering=False,
        description="HP affordance, with filter, GE30",
    ),
    "4b": SetupSpec(
        "4b",
        UNSEEN_CSV,
        hp_affordance=True,
        no_filtering=False,
        description="HP affordance, with filter, unseen",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_summary_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_summary_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def _upsert_row(rows: list[dict], new_row: dict) -> list[dict]:
    key = (str(new_row["round"]), str(new_row["setup_id"]))
    out = [r for r in rows if (str(r.get("round")), str(r.get("setup_id"))) != key]
    out.append(new_row)
    def _sort_key(r: dict) -> tuple[int, int]:
        sid = str(r.get("setup_id", ""))
        try:
            order = SETUP_ORDER.index(sid)
        except ValueError:
            order = 999
        return int(r["round"]), order

    out.sort(key=_sort_key)
    return out


def _object_rate_buckets(by_object: dict) -> dict[str, int]:
    """Per-object success_rate tiers (over trials_per_obj_yaw episodes)."""
    buckets = {
        "n_obj_success_rate_1.0": 0,
        "n_obj_success_rate_ge_0.8": 0,
        "n_obj_success_rate_ge_0.6": 0,
        "n_obj_success_rate_ge_0.4": 0,
        "n_obj_success_rate_eq_0": 0,
    }
    for stats in by_object.values():
        if not isinstance(stats, dict):
            continue
        rate = float(stats.get("success_rate", 0.0))
        total = int(stats.get("total", 0))
        if total <= 0:
            continue
        if rate >= 1.0 - 1e-9:
            buckets["n_obj_success_rate_1.0"] += 1
        if rate >= 0.8:
            buckets["n_obj_success_rate_ge_0.8"] += 1
        if rate >= 0.6:
            buckets["n_obj_success_rate_ge_0.6"] += 1
        if rate >= 0.4:
            buckets["n_obj_success_rate_ge_0.4"] += 1
        if rate <= 1e-9:
            buckets["n_obj_success_rate_eq_0"] += 1
    return buckets


def _metrics_from_eval_summary(result_dir: Path) -> dict:
    path = result_dir / "eval_summary.json"
    if not path.is_file():
        return {
            "n_objects": "",
            "n_episodes": "",
            "n_success": "",
            "success_rate": "",
            "failure_stages_json": "",
            **_object_rate_buckets({}),
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    success = data.get("success") or {}
    counts = data.get("counts") or {}
    by_object = data.get("by_object") or {}
    buckets = _object_rate_buckets(by_object)
    failure = data.get("failure_stages") or {}
    return {
        "n_objects": counts.get("n_objects", len(by_object)),
        "n_episodes": success.get("n_total", counts.get("n_tasks", "")),
        "n_success": success.get("n_success", ""),
        "success_rate": success.get("success_rate", ""),
        "failure_stages_json": json.dumps(failure, sort_keys=True),
        **buckets,
    }


def _result_dir(root: Path, round_idx: int, setup_id: str) -> Path:
    return root / f"{round_idx:03d}_{setup_id}"


def _should_skip(result_dir: Path, resume: bool) -> bool:
    return resume and (result_dir / "eval_summary.json").is_file()


def _build_eval_pool_cmd(
    *,
    spec: SetupSpec,
    result_dir: Path,
    trials: int,
    candidate_gpu_ids: str,
    candidate_workers: int | None,
    candidate_per_gpu: int | None,
    sim_gpu_ids: str,
    sim_per_gpu: int,
    loud: bool,
    resume: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJ / "evaluation" / "eval_pool.py"),
        "--obj-list",
        str(spec.obj_list),
        "--result-dir",
        str(result_dir),
        "--generate-candidate-each-trial",
        "--trials-per-obj-yaw",
        str(int(trials)),
        "--z-yaw-deg",
        "0",
        "--selection",
        "sample",
        "--candidate-gpu-ids",
        candidate_gpu_ids,
        "--sim-gpu-ids",
        sim_gpu_ids,
        "--sim-per-gpu",
        str(int(sim_per_gpu)),
        "--headless",
        "--save-hdf5",
    ]
    if spec.hp_affordance:
        cmd.append("--hp-affordance")
    if spec.no_filtering:
        cmd.append("--no-filtering")
    if candidate_workers is not None:
        cmd.extend(["--candidate-workers", str(int(candidate_workers))])
    elif candidate_per_gpu is not None:
        cmd.extend(["--candidate-per-gpu", str(int(candidate_per_gpu))])
    if resume:
        cmd.append("--resume")
    if loud:
        cmd.append("--loud")
    else:
        cmd.append("--log-only")
    return cmd


def _run_one(
    *,
    round_idx: int,
    spec: SetupSpec,
    root: Path,
    args: argparse.Namespace,
) -> dict:
    result_dir = _result_dir(root, round_idx, spec.setup_id)
    started = _utc_now()
    t0 = time.perf_counter()

    row: dict = {
        "round": round_idx,
        "setup_id": spec.setup_id,
        "status": "pending",
        "result_dir": str(result_dir),
        "obj_list": str(spec.obj_list.relative_to(PROJ)),
        "hp_affordance": int(spec.hp_affordance),
        "filtering": int(not spec.no_filtering),
        "trials_per_obj_yaw": int(args.trials_per_obj_yaw),
        "started_at": started,
        "finished_at": "",
        "duration_s": "",
        "returncode": "",
        "notes": spec.description,
    }

    if _should_skip(result_dir, bool(args.resume)):
        row["status"] = "skipped_resume"
        row["finished_at"] = _utc_now()
        row["duration_s"] = 0.0
        row["returncode"] = 0
        row.update(_metrics_from_eval_summary(result_dir))
        return row

    if args.dry_run:
        cmd = _build_eval_pool_cmd(
            spec=spec,
            result_dir=result_dir,
            trials=args.trials_per_obj_yaw,
            candidate_gpu_ids=args.candidate_gpu_ids,
            candidate_workers=args.candidate_workers,
            candidate_per_gpu=args.candidate_per_gpu,
            sim_gpu_ids=args.sim_gpu_ids,
            sim_per_gpu=args.sim_per_gpu,
            loud=bool(args.loud),
            resume=bool(args.resume),
        )
        row["status"] = "dry_run"
        row["notes"] = " ".join(cmd)
        row["finished_at"] = _utc_now()
        return row

    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_eval_pool_cmd(
        spec=spec,
        result_dir=result_dir,
        trials=args.trials_per_obj_yaw,
        candidate_gpu_ids=args.candidate_gpu_ids,
        candidate_workers=args.candidate_workers,
        candidate_per_gpu=args.candidate_per_gpu,
        sim_gpu_ids=args.sim_gpu_ids,
        sim_per_gpu=args.sim_per_gpu,
        loud=bool(args.loud),
        resume=bool(args.resume),
    )
    print(f"\n{'=' * 72}", flush=True)
    print(f"[round-eval] round={round_idx:03d} setup={spec.setup_id} -> {result_dir}", flush=True)
    print("[round-eval] " + " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd=str(PROJ))
    elapsed = time.perf_counter() - t0
    row["finished_at"] = _utc_now()
    row["duration_s"] = f"{elapsed:.1f}"
    row["returncode"] = proc.returncode
    row["status"] = "ok" if proc.returncode == 0 else "failed"
    row.update(_metrics_from_eval_summary(result_dir))
    if proc.returncode != 0:
        row["notes"] = f"{spec.description}; eval_pool exit {proc.returncode}"
    return row


def _parse_setups(text: str | None) -> list[str]:
    """Return setup ids in canonical order (1a … 4b)."""
    if not text:
        return list(SETUP_ORDER)
    requested = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in requested if s not in SETUPS]
    if bad:
        raise SystemExit(f"Unknown setup id(s): {bad}; valid: {list(SETUP_ORDER)}")
    wanted = set(requested)
    return [sid for sid in SETUP_ORDER if sid in wanted]


def _apply_start_setup(setup_ids: list[str], start_setup: str | None) -> list[str]:
    """Keep only setups from start_setup onward (within the selected list)."""
    if not start_setup:
        return setup_ids
    sid = str(start_setup).strip()
    if sid not in SETUPS:
        raise SystemExit(f"Unknown --start-setup {sid!r}; valid: {list(SETUP_ORDER)}")
    if sid not in setup_ids:
        raise SystemExit(
            f"--start-setup {sid!r} is not in the active setup list {setup_ids}. "
            "Include it in --setups or omit --setups to run all."
        )
    return setup_ids[setup_ids.index(sid) :]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-round eval_pool runner for setups 1a–4b with CSV summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Number of rounds (each round runs all selected setups).",
    )
    p.add_argument("--start-round", type=int, default=1, help="First round index (1-based).")
    p.add_argument(
        "--setups",
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated subset to run, e.g. 2a,4b (order ignored; always 1a→4b). "
            "Default: all eight setups."
        ),
    )
    p.add_argument(
        "--start-setup",
        default=None,
        metavar="ID",
        help=(
            "Start each round from this setup onward (inclusive), e.g. 2a runs 2a…4b "
            "when --setups is omitted. Must appear in the active --setups list."
        ),
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Output root directory.")
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=f"Summary CSV path (default: <root>/{DEFAULT_SUMMARY_CSV}).",
    )
    p.add_argument("--trials-per-obj-yaw", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--candidate-gpu-ids", default="0,1")
    p.add_argument("--candidate-workers", type=int, default=6)
    p.add_argument("--candidate-per-gpu", type=int, default=None)
    p.add_argument("--sim-gpu-ids", default="0,1")
    p.add_argument("--sim-per-gpu", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="Skip setup if eval_summary.json exists.")
    p.add_argument("--dry-run", action="store_true", help="Print eval_pool commands only.")
    p.add_argument(
        "--loud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="eval_pool prints worker output and writes logs (default: on). Use --no-loud for --log-only.",
    )
    p.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop remaining setups/rounds if eval_pool fails.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.start_round < 1:
        raise SystemExit("--start-round must be >= 1")

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv
        else (root / DEFAULT_SUMMARY_CSV)
    )
    setup_ids = _apply_start_setup(_parse_setups(args.setups), args.start_setup)
    if not setup_ids:
        raise SystemExit("No setups selected after --setups / --start-setup.")
    rows = _load_summary_rows(summary_path)

    end_round = args.start_round + int(args.rounds) - 1
    print(
        f"[round-eval] root={root}\n"
        f"[round-eval] rounds {args.start_round:03d}..{end_round:03d} "
        f"setups={','.join(setup_ids)} trials/obj/yaw={args.trials_per_obj_yaw}",
        flush=True,
    )

    for round_idx in range(args.start_round, end_round + 1):
        print(f"\n[round-eval] ===== ROUND {round_idx:03d} =====", flush=True)
        for setup_id in setup_ids:
            spec = SETUPS[setup_id]
            row = _run_one(round_idx=round_idx, spec=spec, root=root, args=args)
            rows = _upsert_row(rows, row)
            _write_summary_rows(summary_path, rows)
            print(
                f"[round-eval] recorded round={round_idx} setup={setup_id} "
                f"status={row['status']} success_rate={row.get('success_rate', '')}",
                flush=True,
            )
            if args.stop_on_failure and row["status"] == "failed":
                raise SystemExit(f"Stopped after failed setup {setup_id} round {round_idx}")

    print(f"\n[round-eval] done -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()

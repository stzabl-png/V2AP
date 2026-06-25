"""
grasp_pool_common.py — 候选池 sim batch 规划 / registry / pool↔round 拷贝
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
from typing import Any, Optional

import h5py
import numpy as np

FIXED_Z_YAWS = (0.0, 90.0, 180.0, 270.0)
DEFAULT_SLOTS_PER_ROUND = 500
REGISTRY_NAME = "sim_pool_registry.json"
TASK_QUEUE_TEMPLATE = "round_{round:04d}_task_queue.json"
SYNTH_SKIPPED_RESULTS_NAME = "synthesized_skipped_results.json"
ACCUMULATED_RESULTS_NAME = "accumulated_results.json"
SKIP_REASON_CANDIDATE_SUCCESS = "candidate_success_at_yaw"


def round_tag(round_idx: int) -> str:
    return f"round_{round_idx:04d}"


def task_queue_name(round_idx: int) -> str:
    return TASK_QUEUE_TEMPLATE.format(round=round_idx)


def paths_for_outdir(outdir: str, round_idx: int) -> dict[str, str]:
    tag = round_tag(round_idx)
    base = os.path.abspath(outdir)
    return {
        "outdir": base,
        "pool_dir": os.path.join(base, "candidates", "pool"),
        "cand_round_dir": os.path.join(base, "candidates", tag),
        "gt_round_dir": os.path.join(base, "robot_gt", tag),
        "merged_dir": os.path.join(base, "merged"),
        "log_dir": os.path.join(base, "sim_logs", tag),
        "registry": os.path.join(base, REGISTRY_NAME),
        "task_queue": os.path.join(base, task_queue_name(round_idx)),
        "state": os.path.join(base, "state.json"),
        "summary": os.path.join(base, "summary.csv"),
    }


def pool_hdf5(pool_dir: str, obj_id: str) -> str:
    return os.path.join(pool_dir, f"{obj_id}_grasp.hdf5")


def round_grasp_hdf5(cand_round_dir: str, obj_id: str) -> str:
    return os.path.join(cand_round_dir, f"{obj_id}_grasp.hdf5")


def round_gt_hdf5(gt_round_dir: str, obj_id: str) -> str:
    return os.path.join(gt_round_dir, f"{obj_id}_robot_gt.hdf5")


def merged_hdf5(merged_dir: str, obj_id: str) -> str:
    return os.path.join(merged_dir, f"{obj_id}_robot_gt_merged.hdf5")


def _parse_round_from_path(path: str) -> Optional[int]:
    m = re.search(r"round_(\d+)", path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def count_success_in_gt_file(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with h5py.File(path, "r") as f:
        if "successful_grasps" in f:
            sg = f["successful_grasps"]
            n_attr = int(sg.attrs.get("count", -1))
            if n_attr >= 0:
                return n_attr
            return len(sg.keys())
        return int(f.attrs.get("n_successful", 0))


def scan_success_round_ge3(outdir: str, min_round: int = 3) -> dict[str, int]:
    """
    Legacy: obj_id -> successful_grasps summed over robot_gt/round_R for R >= min_round.

    Pool sim planning, auto-refill threshold, and --max-success-per-object use
    scan_merged_objects() instead. Kept for ad-hoc scripts / --full-merge tooling.
    """
    totals: dict[str, int] = {}
    gt_root = os.path.join(os.path.abspath(outdir), "robot_gt")
    if not os.path.isdir(gt_root):
        return totals
    for tag in sorted(os.listdir(gt_root)):
        if not tag.startswith("round_"):
            continue
        r = _parse_round_from_path(tag)
        if r is None or r < min_round:
            continue
        rd = os.path.join(gt_root, tag)
        for fn in os.listdir(rd):
            if not fn.endswith("_robot_gt.hdf5"):
                continue
            obj_id = fn[: -len("_robot_gt.hdf5")]
            n = count_success_in_gt_file(os.path.join(rd, fn))
            totals[obj_id] = totals.get(obj_id, 0) + n
    return totals


def compute_median_success_threshold(merged_dir: str) -> int:
    """
    Median of n_successful in merged/ over objects that have a merged file.
    Ad-hoc / scripts; pool sim auto-refill uses --pool-success-threshold (default 20).
    """
    merged_counts = scan_merged_objects(merged_dir)
    values = list(merged_counts.values())
    if not values:
        return 0
    return int(np.median(np.array(values, dtype=np.float64)))


def all_merged_objects_at_success_cap(merged_dir: str, cap: int) -> bool:
    """True if every object in merged/ has n_successful >= cap."""
    counts = scan_merged_objects(merged_dir)
    if not counts:
        return False
    return all(n >= cap for n in counts.values())


def scan_merged_objects(merged_dir: str) -> dict[str, int]:
    from merge_robot_gt import _count_successful_in_merged

    counts: dict[str, int] = {}
    pattern = os.path.join(merged_dir, "*_robot_gt_merged.hdf5")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        obj_id = base[: -len("_robot_gt_merged.hdf5")]
        counts[obj_id] = _count_successful_in_merged(path)
    return counts


def is_trusted_grasp_group(g: h5py.Group) -> bool:
    """Same rules as prepare_affordance_executed.is_trusted_grasp."""
    if "gripper_tips_loc" not in g:
        return False
    if bool(g.attrs.get("gripper_tips_trusted", False)):
        return True
    if str(g.attrs.get("gripper_tips_source", "")) == "legacy_post_lift":
        return False
    return str(g.attrs.get("gripper_tips_snapshot", "at_close")) == "at_close"


def count_trusted_grasps_in_merged(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with h5py.File(path, "r") as f:
        if "successful_grasps" not in f:
            return 0
        grp = f["successful_grasps"]
        return sum(1 for key in grp.keys() if is_trusted_grasp_group(grp[key]))


def scan_merged_trusted_objects(merged_dir: str) -> dict[str, int]:
    """obj_id -> trusted successful_grasps count in merged/{obj}_robot_gt_merged.hdf5."""
    counts: dict[str, int] = {}
    pattern = os.path.join(os.path.abspath(merged_dir), "*_robot_gt_merged.hdf5")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        obj_id = base[: -len("_robot_gt_merged.hdf5")]
        counts[obj_id] = count_trusted_grasps_in_merged(path)
    return counts


def load_registry(path: str) -> dict:
    if not os.path.isfile(path):
        return {"version": 1, "candidates": {}}
    with open(path, "r") as f:
        data = json.load(f)
    data.setdefault("candidates", {})
    return data


def save_registry(path: str, registry: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def clear_registry_for_objects(registry: dict, obj_ids: list[str]) -> int:
    """Remove sim history for objects whose pool HDF5 was regenerated. Returns count cleared."""
    cand = registry.setdefault("candidates", {})
    n = 0
    for obj_id in obj_ids:
        if obj_id in cand:
            del cand[obj_id]
            n += 1
    return n


def _obj_registry(registry: dict, obj_id: str) -> dict:
    return registry.setdefault("candidates", {}).setdefault(obj_id, {})


def candidate_key(name: str, pool_idx: int) -> str:
    return f"{name}#{pool_idx}"


def parse_candidate_key(key: str) -> tuple[str, int]:
    if "#" in key:
        name, idx_s = key.rsplit("#", 1)
        return name, int(idx_s)
    return key, -1


def candidate_pair_from_row(row: dict) -> tuple[str, str]:
    return (
        row["obj_id"],
        row.get("candidate_key") or row.get("candidate_name", ""),
    )


def candidate_pair_from_task(task: dict) -> tuple[str, str]:
    return (task["obj_id"], task.get("candidate_key", task.get("candidate_name", "")))


def make_skipped_result(task: dict, success_yaw_deg: float) -> dict:
    """Synthetic chunk row: sibling yaw succeeded, this yaw was not simmed."""
    return {
        "task_id": task["task_id"],
        "obj_id": task["obj_id"],
        "candidate_key": task.get("candidate_key", task.get("candidate_name", "")),
        "candidate_name": task.get("candidate_name", ""),
        "z_yaw_deg": float(task["z_yaw_deg"]),
        "success": False,
        "attempted": False,
        "skipped": True,
        "skip_reason": SKIP_REASON_CANDIDATE_SUCCESS,
        "success_yaw_deg": float(success_yaw_deg),
        "error": "",
    }


def find_candidate_success_yaw(rows: list[dict]) -> Optional[float]:
    for row in rows:
        if row.get("success"):
            return float(row.get("z_yaw_deg", 0.0))
    return None


def synthesize_skipped_results_for_queue(
    queue: dict,
    round_results: list[dict],
) -> list[dict]:
    """Skipped rows for queue tasks whose candidate already has a success in round_results."""
    by_tid = {r["task_id"]: r for r in round_results if r.get("task_id")}
    by_cand: dict[tuple[str, str], list[dict]] = {}
    for row in round_results:
        by_cand.setdefault(candidate_pair_from_row(row), []).append(row)

    synth: list[dict] = []
    for task in queue.get("tasks", []):
        tid = task["task_id"]
        if tid in by_tid:
            continue
        pair = candidate_pair_from_task(task)
        success_yaw = find_candidate_success_yaw(by_cand.get(pair, []))
        if success_yaw is not None:
            synth.append(make_skipped_result(task, success_yaw))
    return synth


def is_fully_simulated(registry: dict, obj_id: str, key: str) -> bool:
    """Candidate resolved: any yaw success, or all 4 yaws tried without success."""
    rec = _obj_registry(registry, obj_id).get(key)
    if not rec:
        return False
    if rec.get("success_yaws"):
        return True
    if rec.get("simulated"):
        return True
    done = set(float(y) for y in rec.get("yaws_done", []))
    return done >= set(FIXED_Z_YAWS)


def list_pool_candidates_sorted(pool_path: str) -> list[dict[str, Any]]:
    """Candidates from pool HDF5, highest score first."""
    if not os.path.isfile(pool_path):
        return []
    out: list[dict[str, Any]] = []
    with h5py.File(pool_path, "r") as f:
        if "candidates" not in f:
            return out
        cg = f["candidates"]
        n = int(cg.attrs.get("n_candidates", 0))
        for i in range(n):
            gname = f"candidate_{i}"
            if gname not in cg:
                continue
            ci = cg[gname]
            name = str(ci.attrs.get("name", gname))
            out.append(
                {
                    "pool_idx": i,
                    "name": name,
                    "key": candidate_key(name, i),
                    "score": float(ci.attrs.get("score", 0.0)),
                }
            )
    out.sort(key=lambda c: (-c["score"], c["pool_idx"]))
    return out


def available_pool_candidates(
    registry: dict,
    pool_path: str,
    obj_id: str,
) -> list[dict[str, Any]]:
    return [
        c
        for c in list_pool_candidates_sorted(pool_path)
        if not is_fully_simulated(registry, obj_id, c["key"])
    ]


def eligible_objects(
    merged_dir: str,
    pool_dir: str,
    registry: dict,
) -> list[str]:
    """Objects with merged file, pool HDF5, and ≥1 unsimulated candidate."""
    merged = scan_merged_objects(merged_dir)
    eligible: list[str] = []
    for obj_id in sorted(merged.keys()):
        pp = pool_hdf5(pool_dir, obj_id)
        if not os.path.isfile(pp):
            continue
        if available_pool_candidates(registry, pp, obj_id):
            eligible.append(obj_id)
    return eligible


def normalize_weights(obj_ids: list[str], success: dict[str, int]) -> dict[str, float]:
    weights = {oid: 1.0 / (success.get(oid, 0) + 1) for oid in obj_ids}
    total = sum(weights.values())
    if total <= 0:
        n = len(obj_ids)
        return {oid: 1.0 / n for oid in obj_ids}
    return {oid: w / total for oid, w in weights.items()}


def normalize_equal_weights(obj_ids: list[str]) -> dict[str, float]:
    if not obj_ids:
        return {}
    p = 1.0 / len(obj_ids)
    return {oid: p for oid in obj_ids}


def object_selection_probs(
    obj_ids: list[str],
    success: dict[str, int],
    *,
    equal_object_prob: bool = False,
    max_success_per_object: Optional[int] = None,
) -> dict[str, float]:
    """success: obj_id -> count (pool batch uses scan_merged_objects)."""
    if equal_object_prob:
        probs = normalize_equal_weights(obj_ids)
    else:
        probs = normalize_weights(obj_ids, success)
    if max_success_per_object is not None:
        for oid in obj_ids:
            if success.get(oid, 0) >= max_success_per_object:
                probs[oid] = 0.0
        total = sum(probs.values())
        if total > 0:
            probs = {oid: w / total for oid, w in probs.items()}
    return probs


def sample_object(
    rng: np.random.Generator,
    obj_ids: list[str],
    probs: dict[str, float],
) -> Optional[str]:
    if not obj_ids:
        return None
    p = np.array([probs.get(oid, 0.0) for oid in obj_ids], dtype=np.float64)
    total = float(p.sum())
    if total <= 0:
        return None
    p /= total
    idx = int(rng.choice(len(obj_ids), p=p))
    return obj_ids[idx]


def plan_round_slots(
    *,
    outdir: str,
    merged_dir: str,
    pool_dir: str,
    registry: dict,
    round_idx: int,
    slots_per_round: int,
    equal_object_prob: bool = False,
    max_success_per_object: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[dict], bool]:
    """
    Plan up to slots_per_round unique (obj, candidate) assignments.
    Returns (slot_list, exhausted) where exhausted=True if stopped early
    because no eligible candidate remained.

    Object success counts for weighted sampling and max_success cap both come from
    merged/{obj}_robot_gt_merged.hdf5 (n_successful). outdir is unused for planning
    (kept for API compatibility with build_task_queue).
    """
    rng = rng or np.random.default_rng()
    success = scan_merged_objects(merged_dir)
    eligible = eligible_objects(merged_dir, pool_dir, registry)
    if not eligible:
        return [], True

    probs = object_selection_probs(
        eligible,
        success,
        equal_object_prob=equal_object_prob,
        max_success_per_object=max_success_per_object,
    )
    used_this_plan: set[tuple[str, str]] = set()
    slots: list[dict] = []
    exhausted = False

    for _ in range(slots_per_round):
        picked = None
        for _try in range(len(eligible) * 50):
            oid = sample_object(rng, eligible, probs)
            if oid is None:
                exhausted = True
                break
            avail = available_pool_candidates(
                registry, pool_hdf5(pool_dir, oid), oid,
            )
            for cand in avail:
                pair = (oid, cand["key"])
                if pair in used_this_plan:
                    continue
                picked = {
                    "obj_id": oid,
                    "candidate_key": cand["key"],
                    "candidate_name": cand["name"],
                    "pool_idx": cand["pool_idx"],
                    "score": cand["score"],
                }
                used_this_plan.add(pair)
                break
            if picked is not None:
                break
        if picked is None:
            exhausted = True
            break
        slots.append(picked)

    return slots, exhausted


def expand_slots_to_tasks(
    slots: list[dict],
    round_idx: int,
    dataset_by_obj: dict[str, str],
) -> list[dict]:
    tasks: list[dict] = []
    for si, slot in enumerate(slots):
        obj_id = slot["obj_id"]
        for yi, yaw in enumerate(FIXED_Z_YAWS):
            tasks.append(
                {
                    "task_id": f"r{round_idx:04d}_s{si:04d}_y{int(yaw):03d}",
                    "round_idx": round_idx,
                    "obj_id": obj_id,
                    "dataset": dataset_by_obj.get(obj_id, ""),
                    "candidate_key": slot["candidate_key"],
                    "candidate_name": slot["candidate_name"],
                    "pool_idx": slot["pool_idx"],
                    "score": slot["score"],
                    "z_yaw_deg": float(yaw),
                    "slot_index": si,
                    "yaw_index": yi,
                    "status": "pending",
                }
            )
    return tasks


def build_task_queue(
    *,
    outdir: str,
    merged_dir: str,
    pool_dir: str,
    registry: dict,
    round_idx: int,
    slots_per_round: int,
    dataset_by_obj: dict[str, str],
    equal_object_prob: bool = False,
    max_success_per_object: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    slots, exhausted = plan_round_slots(
        outdir=outdir,
        merged_dir=merged_dir,
        pool_dir=pool_dir,
        registry=registry,
        round_idx=round_idx,
        slots_per_round=slots_per_round,
        equal_object_prob=equal_object_prob,
        max_success_per_object=max_success_per_object,
        rng=rng,
    )
    tasks = expand_slots_to_tasks(slots, round_idx, dataset_by_obj)
    queue: dict = {
        "version": 1,
        "round_idx": round_idx,
        "slots_planned": len(slots),
        "slots_target": slots_per_round,
        "pool_exhausted": exhausted,
        "object_sampling": "equal" if equal_object_prob else "weighted",
        "success_source": "merged",
        "tasks": tasks,
        "completed_task_ids": [],
    }
    if max_success_per_object is not None:
        queue["max_success_per_object"] = max_success_per_object
    return queue


def load_task_queue(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_task_queue(path: str, queue: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(queue, f, indent=2)


def pending_tasks(
    queue: dict,
    outdir: str | None = None,
    round_idx: int | None = None,
    *,
    registry: dict | None = None,
    registry_path: str | None = None,
    task_queue_path: str | None = None,
    persist: bool = False,
) -> list[dict]:
    """Pending = in queue plan but no chunk result. Optionally sync from chunk first."""
    if outdir is not None and round_idx is not None and registry is not None:
        sync_queue_and_registry_from_chunks(
            outdir,
            round_idx,
            registry,
            queue,
            registry_path=registry_path,
            task_queue_path=task_queue_path,
            persist=persist,
        )
    done = set(queue.get("completed_task_ids", []))
    return [t for t in queue.get("tasks", []) if t["task_id"] not in done]


def is_queue_complete(
    queue: dict,
    outdir: str | None = None,
    round_idx: int | None = None,
    *,
    registry: dict | None = None,
    registry_path: str | None = None,
    task_queue_path: str | None = None,
    persist: bool = False,
) -> bool:
    """True when every planned task has a chunk result (sim fully attempted on disk)."""
    return (
        len(
            pending_tasks(
                queue,
                outdir,
                round_idx,
                registry=registry,
                registry_path=registry_path,
                task_queue_path=task_queue_path,
                persist=persist,
            ),
        )
        == 0
    )


def mark_task_done(queue: dict, task_id: str) -> None:
    done = queue.setdefault("completed_task_ids", [])
    if task_id not in done:
        done.append(task_id)


def chunk_results_dir(outdir: str, round_idx: int) -> str:
    return os.path.join(os.path.abspath(outdir), "sim_logs", round_tag(round_idx), "chunks")


def synthesized_skipped_results_path(outdir: str, round_idx: int) -> str:
    return os.path.join(chunk_results_dir(outdir, round_idx), SYNTH_SKIPPED_RESULTS_NAME)


def accumulated_results_path(outdir: str, round_idx: int) -> str:
    return os.path.join(chunk_results_dir(outdir, round_idx), ACCUMULATED_RESULTS_NAME)


def persist_accumulated_results(
    outdir: str,
    round_idx: int,
    rows: list[dict],
) -> None:
    """Merge task results into round-level archive (survives chunk sidecar wipe on resume)."""
    if not rows:
        return
    path = accumulated_results_path(outdir, round_idx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing: dict[str, dict] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for row in data.get("results", []):
                tid = row.get("task_id")
                if tid:
                    existing[tid] = row
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            existing = {}
    for row in rows:
        tid = row.get("task_id")
        if tid:
            existing[tid] = row
    with open(path, "w") as f:
        json.dump({"results": list(existing.values())}, f, indent=2)


def archive_chunk_results_before_reshuffle(outdir: str, round_idx: int) -> int:
    """
    Before re-splitting pending tasks into new chunk_*.json, persist all on-disk
    results (chunk sidecars + synth + prior archive) into accumulated_results.json.
    """
    results = load_disk_sim_results(outdir, round_idx)
    if results:
        persist_accumulated_results(outdir, round_idx, results)
    path = accumulated_results_path(outdir, round_idx)
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r") as f:
            return len(json.load(f).get("results", []))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _merge_result_rows_into_by_task(
    by_task: dict[str, tuple[float, dict]],
    rows: list[dict],
    mtime: float,
) -> None:
    for row in rows:
        tid = row.get("task_id")
        if not tid:
            continue
        prev = by_task.get(tid)
        if prev is None or mtime >= prev[0]:
            by_task[tid] = (mtime, row)


def persist_synthesized_skipped_results(
    outdir: str,
    round_idx: int,
    synth_rows: list[dict],
) -> None:
    """Append synthesized skipped rows (sync/resume) to round synth sidecar."""
    if not synth_rows:
        return
    path = synthesized_skipped_results_path(outdir, round_idx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing: dict[str, dict] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for row in data.get("results", []):
                tid = row.get("task_id")
                if tid:
                    existing[tid] = row
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            existing = {}
    for row in synth_rows:
        tid = row.get("task_id")
        if tid:
            existing[tid] = row
    with open(path, "w") as f:
        json.dump({"results": list(existing.values())}, f, indent=2)


def load_disk_sim_results(
    outdir: str,
    round_idx: int,
    *,
    failed_paths: list[str] | None = None,
) -> list[dict]:
    """Merge chunk_*_results.json; newest file mtime wins on duplicate task_id."""
    results, _ = load_disk_sim_results_ex(outdir, round_idx, failed_paths=failed_paths)
    return results


def load_disk_sim_results_ex(
    outdir: str,
    round_idx: int,
    *,
    failed_paths: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Like load_disk_sim_results; also returns chunk result paths that failed to parse."""
    chunk_dir = chunk_results_dir(outdir, round_idx)
    by_task: dict[str, tuple[float, dict]] = {}
    failed: list[str] = []
    acc_path = accumulated_results_path(outdir, round_idx)
    if os.path.isfile(acc_path):
        try:
            mtime = os.path.getmtime(acc_path)
            with open(acc_path, "r") as f:
                data = json.load(f)
            _merge_result_rows_into_by_task(by_task, data.get("results", []), mtime)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            failed.append(acc_path)
    for path in sorted(glob.glob(os.path.join(chunk_dir, "chunk_*_results.json"))):
        try:
            mtime = os.path.getmtime(path)
            with open(path, "r") as f:
                data = json.load(f)
            _merge_result_rows_into_by_task(by_task, data.get("results", []), mtime)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            failed.append(path)
            continue
    synth_path = synthesized_skipped_results_path(outdir, round_idx)
    if os.path.isfile(synth_path):
        try:
            mtime = os.path.getmtime(synth_path)
            with open(synth_path, "r") as f:
                data = json.load(f)
            _merge_result_rows_into_by_task(by_task, data.get("results", []), mtime)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            failed.append(synth_path)
    if failed_paths is not None:
        failed_paths.extend(failed)
    return [pair[1] for pair in by_task.values()], failed


def derive_completed_task_ids_from_chunks(
    outdir: str,
    round_idx: int,
    queue: dict,
    *,
    failed_paths: list[str] | None = None,
) -> set[str]:
    """Task IDs in this round's chunk results that belong to queue.tasks."""
    valid_ids = {t["task_id"] for t in queue.get("tasks", [])}
    results = load_disk_sim_results(
        outdir, round_idx, failed_paths=failed_paths,
    )
    return {r["task_id"] for r in results if r.get("task_id") in valid_ids}


def rebuild_completed_task_ids_from_chunks(queue: dict, done_ids: set[str]) -> bool:
    """Rewrite completed_task_ids cache from chunk truth (plan task order)."""
    new_done = [t["task_id"] for t in queue.get("tasks", []) if t["task_id"] in done_ids]
    old = queue.get("completed_task_ids", [])
    changed = old != new_done
    queue["completed_task_ids"] = new_done
    return changed


def apply_sim_results(
    registry: dict,
    queue: dict,
    results: list[dict],
    round_idx: int,
) -> int:
    """Legacy incremental path: registry update + append completed ids."""
    valid_ids = {t["task_id"] for t in queue.get("tasks", [])}
    done = set(queue.get("completed_task_ids", []))
    new_results = [
        r for r in results
        if r.get("task_id") in valid_ids and r["task_id"] not in done
    ]
    if not new_results:
        return 0
    update_registry_from_results(registry, new_results, round_idx)
    for row in new_results:
        mark_task_done(queue, row["task_id"])
    return len(new_results)


def sync_queue_and_registry_from_chunks(
    outdir: str,
    round_idx: int,
    registry: dict,
    queue: dict,
    *,
    registry_path: str | None = None,
    task_queue_path: str | None = None,
    persist: bool = True,
) -> dict:
    """
    Chunk results are the source of truth: rebuild completed_task_ids and update registry.

    Returns status with sync_ok (no unread chunk files; queue matches chunk) and sim_complete.
    """
    valid_ids = {t["task_id"] for t in queue.get("tasks", [])}
    failed_paths: list[str] = []
    results = load_disk_sim_results(outdir, round_idx, failed_paths=failed_paths)
    round_results = [r for r in results if r.get("task_id") in valid_ids]
    synth_rows = synthesize_skipped_results_for_queue(queue, round_results)
    existing_tids = {r["task_id"] for r in round_results}
    new_synth = [r for r in synth_rows if r["task_id"] not in existing_tids]
    if new_synth and persist:
        persist_synthesized_skipped_results(outdir, round_idx, new_synth)
    round_results.extend(new_synth)
    chunk_ids = {r["task_id"] for r in round_results}

    old_done = set(queue.get("completed_task_ids", []))
    registry_changed = update_registry_from_results(registry, round_results, round_idx)
    queue_changed = rebuild_completed_task_ids_from_chunks(queue, chunk_ids)
    n_newly_completed = len(chunk_ids - old_done)

    done_set = set(queue["completed_task_ids"])
    ingest_gap = chunk_ids - done_set
    sim_missing = valid_ids - chunk_ids
    sync_ok = len(ingest_gap) == 0 and len(failed_paths) == 0

    if persist and (queue_changed or registry_changed or new_synth):
        if registry_path:
            save_registry(registry_path, registry)
        if task_queue_path:
            save_task_queue(task_queue_path, queue)
    if persist and round_results:
        persist_accumulated_results(outdir, round_idx, round_results)

    n_pending = len([t for t in queue.get("tasks", []) if t["task_id"] not in done_set])
    return {
        "n_chunk_tasks": len(chunk_ids),
        "n_completed": len(done_set),
        "n_pending": n_pending,
        "n_ingest_gap": len(ingest_gap),
        "n_sim_missing": len(sim_missing),
        "n_newly_completed": n_newly_completed,
        "n_failed_chunk_files": len(failed_paths),
        "failed_chunk_files": failed_paths,
        "sync_ok": sync_ok,
        "sim_complete": n_pending == 0,
        "queue_changed": queue_changed,
        "registry_changed": registry_changed,
        "n_synthesized_skipped": len(new_synth),
    }


def ingest_and_persist_sim_progress(
    outdir: str,
    round_idx: int,
    registry: dict,
    queue: dict,
    *,
    registry_path: str,
    task_queue_path: str,
) -> int:
    """Sync queue/registry from chunk results; return count of newly completed tasks."""
    status = sync_queue_and_registry_from_chunks(
        outdir,
        round_idx,
        registry,
        queue,
        registry_path=registry_path,
        task_queue_path=task_queue_path,
        persist=True,
    )
    return int(status.get("n_newly_completed", 0))


def _copy_candidate_group(src_ci: h5py.Group, dst_cg: h5py.Group, dst_index: int) -> None:
    gname = f"candidate_{dst_index}"
    if gname in dst_cg:
        del dst_cg[gname]
    dst = dst_cg.create_group(gname)
    for key in src_ci.keys():
        src_ci.copy(src_ci[key], dst, key)
    for ak, av in src_ci.attrs.items():
        dst.attrs[ak] = av


def copy_slots_to_round_hdf5(
    pool_dir: str,
    cand_round_dir: str,
    slots: list[dict],
) -> dict[str, int]:
    """
    Merge this round's pool candidates into per-obj round grasp HDF5.
    Returns obj_id -> n_candidates in round file.
    """
    os.makedirs(cand_round_dir, exist_ok=True)
    by_obj: dict[str, list[dict]] = {}
    for slot in slots:
        by_obj.setdefault(slot["obj_id"], []).append(slot)

    counts: dict[str, int] = {}
    for obj_id, obj_slots in by_obj.items():
        pool_path = pool_hdf5(pool_dir, obj_id)
        round_path = round_grasp_hdf5(cand_round_dir, obj_id)
        keys_needed = {s["candidate_key"] for s in obj_slots}

        existing: dict[str, int] = {}
        next_idx = 0
        if os.path.isfile(round_path):
            with h5py.File(round_path, "r") as rf:
                if "candidates" in rf:
                    cg = rf["candidates"]
                    n = int(cg.attrs.get("n_candidates", 0))
                    for i in range(n):
                        gname = f"candidate_{i}"
                        if gname not in cg:
                            continue
                        ci = cg[gname]
                        name = str(ci.attrs.get("name", gname))
                        key = candidate_key(name, i)
                        existing[key] = i
                    next_idx = n

        with h5py.File(pool_path, "r") as pf:
            if "metadata" in pf:
                meta_attrs = dict(pf["metadata"].attrs.items())
            else:
                meta_attrs = dict(pf.attrs.items())
            pool_cg = pf["candidates"]

            mode = "a" if os.path.isfile(round_path) else "w"
            with h5py.File(round_path, mode) as wf:
                if "metadata" not in wf:
                    mg = wf.create_group("metadata")
                    for k, v in meta_attrs.items():
                        mg.attrs[k] = v
                if "candidates" not in wf:
                    wf.create_group("candidates")
                dst_cg = wf["candidates"]

                for slot in obj_slots:
                    key = slot["candidate_key"]
                    if key in existing:
                        continue
                    pi = slot["pool_idx"]
                    src_name = f"candidate_{pi}"
                    if src_name not in pool_cg:
                        continue
                    _copy_candidate_group(pool_cg[src_name], dst_cg, next_idx)
                    dst_cg[f"candidate_{next_idx}"].attrs["pool_candidate_key"] = key
                    dst_cg[f"candidate_{next_idx}"].attrs["pool_idx"] = pi
                    existing[key] = next_idx
                    next_idx += 1

                dst_cg.attrs["n_candidates"] = next_idx
                counts[obj_id] = next_idx

    return counts


def unique_slots_from_tasks(tasks: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    slots: list[dict] = []
    for t in tasks:
        pair = (t["obj_id"], t["candidate_key"])
        if pair in seen:
            continue
        seen.add(pair)
        slots.append(
            {
                "obj_id": t["obj_id"],
                "candidate_key": t["candidate_key"],
                "candidate_name": t["candidate_name"],
                "pool_idx": t["pool_idx"],
                "score": t.get("score", 0.0),
            }
        )
    return slots


def update_registry_from_results(
    registry: dict,
    results: list[dict],
    round_idx: int,
) -> bool:
    """Apply chunk rows to registry (idempotent). Returns True if any record changed."""
    by_cand: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        pair = (r["obj_id"], r["candidate_key"])
        by_cand.setdefault(pair, []).append(r)

    changed = False
    for (obj_id, key), rows in by_cand.items():
        rec = _obj_registry(registry, obj_id).setdefault(
            key,
            {"yaws_done": [], "simulated": False},
        )
        before = (list(rec.get("yaws_done", [])), rec.get("simulated"), rec.get("last_round"))
        for row in rows:
            yaw = float(row.get("z_yaw_deg", 0.0))
            done = set(float(y) for y in rec.get("yaws_done", []))
            done.add(yaw)
            rec["yaws_done"] = sorted(done)
            if row.get("success"):
                success_yaws = rec.setdefault("success_yaws", [])
                if yaw not in {float(y) for y in success_yaws}:
                    success_yaws.append(yaw)
        rec["last_round"] = round_idx
        if rec.get("success_yaws"):
            rec["simulated"] = True
        else:
            yaws_done_set = {float(y) for y in rec.get("yaws_done", [])}
            if yaws_done_set >= set(FIXED_Z_YAWS):
                rec["simulated"] = True
        after = (list(rec.get("yaws_done", [])), rec.get("simulated"), rec.get("last_round"))
        if before != after:
            changed = True
    return changed


def sort_tasks_for_workers(tasks: list[dict]) -> list[dict]:
    return sorted(
        tasks,
        key=lambda t: (t["obj_id"], t.get("slot_index", 0), t.get("yaw_index", 0)),
    )


def _task_groups_by_candidate(tasks: list[dict]) -> list[list[dict]]:
    """Keep all yaws for one (obj, candidate) in the same group."""
    tasks = sort_tasks_for_workers(tasks)
    if not tasks:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = []
    last_pair: tuple[str, str] | None = None
    for task in tasks:
        pair = candidate_pair_from_task(task)
        if last_pair is not None and pair != last_pair and current:
            groups.append(current)
            current = []
        current.append(task)
        last_pair = pair
    if current:
        groups.append(current)
    return groups


def split_tasks_into_chunks(tasks: list[dict], n_chunks: int) -> list[list[dict]]:
    """Split pending tasks into worker chunks without splitting a candidate yaw group."""
    if n_chunks < 1:
        n_chunks = 1
    groups = _task_groups_by_candidate(tasks)
    if not groups:
        return []
    n_chunks = min(n_chunks, len(groups))
    chunks: list[list[dict]] = [[] for _ in range(n_chunks)]
    loads = [0] * n_chunks
    # assign largest groups first to the lightest chunk
    indexed = sorted(enumerate(groups), key=lambda item: -len(item[1]))
    for _idx, group in indexed:
        ci = min(range(n_chunks), key=lambda i: loads[i])
        chunks[ci].extend(group)
        loads[ci] += len(group)
    return [sort_tasks_for_workers(c) for c in chunks if c]

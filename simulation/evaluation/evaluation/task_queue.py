"""Task queue and chunk helpers for evaluation pool workers."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_task_queue(solution_manifest: dict, *, sort_by_object: bool = True) -> dict:
    rows = list(solution_manifest.get("solutions", []))
    if sort_by_object:
        rows.sort(key=lambda r: (str(r["obj_id"]), float(r["z_yaw_deg"]), int(r["trial"])))
    tasks = []
    for idx, row in enumerate(rows):
        tasks.append(
            {
                "task_id": row["episode_id"],
                "task_index": idx,
                "solution_id": row["solution_id"],
                "episode_id": row["episode_id"],
                "obj_id": row["obj_id"],
                "trial": int(row["trial"]),
                "z_yaw_deg": float(row["z_yaw_deg"]),
                "solution_path": row["solution_path"],
                "record_video": bool(row.get("record_video", False)),
                "status": "pending",
            }
        )
    return {
        "version": 1,
        "n_tasks": len(tasks),
        "tasks": tasks,
        "completed_task_ids": [],
    }


def split_tasks(tasks: list[dict], n_chunks: int) -> list[list[dict]]:
    """Split tasks by object groups, greedily balancing task counts.

    Keeping all trials/yaws for one object in the same chunk reduces object
    swaps inside a long-lived worker. Sorting groups by size first keeps the
    chunk lengths reasonably balanced when object group sizes differ.
    """
    n_chunks = max(1, int(n_chunks))
    chunks = [[] for _ in range(n_chunks)]

    groups: dict[str, list[dict]] = {}
    for task in tasks:
        groups.setdefault(str(task["obj_id"]), []).append(task)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    chunk_sizes = [0 for _ in range(n_chunks)]
    for _obj_id, group in ordered_groups:
        chunk_idx = min(range(n_chunks), key=lambda idx: (chunk_sizes[idx], idx))
        chunks[chunk_idx].extend(group)
        chunk_sizes[chunk_idx] += len(group)

    return [c for c in chunks if c]


def write_chunks(
    *,
    queue: dict,
    result_dir: Path,
    n_chunks: int,
    object_scale: float,
    headless: bool,
    save_hdf5: bool,
    record_fps: int,
    record_every: int,
    record_keep_frames: bool,
) -> list[Path]:
    chunk_dir = result_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ci, tasks in enumerate(split_tasks(queue.get("tasks", []), n_chunks)):
        path = chunk_dir / f"chunk_{ci:03d}.json"
        write_json(
            path,
            {
                "version": 1,
                "chunk_id": f"chunk_{ci:03d}",
                "result_dir": str(result_dir.resolve()),
                "object_scale": float(object_scale),
                "headless": bool(headless),
                "save_hdf5": bool(save_hdf5),
                "record_fps": int(record_fps),
                "record_every": int(record_every),
                "record_keep_frames": bool(record_keep_frames),
                "tasks": tasks,
            },
        )
        paths.append(path)
    return paths


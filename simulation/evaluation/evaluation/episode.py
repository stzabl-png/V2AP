"""Episode job discovery and scheduling helpers for batch evaluation."""

from __future__ import annotations

import os
import csv
import json
from dataclasses import dataclass
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EpisodeJob:
    obj_id: str
    trial: int
    z_yaw_deg: float
    episode_id: str
    candidate_hdf5: str | None = None
    record_video: bool = False


def obj_id_from_usd(path: Path) -> str:
    return path.stem


def discover_obj_ids(
    *,
    usd_root: str | Path | None = None,
    candidate_dir: str | Path | None = None,
    obj_ids: list[str] | None = None,
    obj_list_file: str | Path | None = None,
) -> list[str]:
    found: list[str] = []

    if obj_ids:
        found.extend(obj_ids)

    if obj_list_file:
        path = Path(obj_list_file).expanduser()
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("objects", data) if isinstance(data, dict) else data
            for row in rows:
                if isinstance(row, str):
                    found.append(row)
                elif isinstance(row, dict) and row.get("enabled", True):
                    oid = row.get("obj_id") or row.get("id") or row.get("object")
                    if oid:
                        found.append(str(oid))
        elif suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
                for row in reader:
                    enabled = str(row.get("enabled", "1")).strip().lower()
                    if enabled in {"0", "false", "no", "n", "off"}:
                        continue
                    oid = row.get("obj_id") or row.get("id") or row.get("object")
                    if oid:
                        found.append(str(oid).strip())
        else:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        found.append(line.split()[0])

    if usd_root:
        root = Path(usd_root).expanduser()
        if root.is_file() and root.suffix.lower() == ".usd":
            found.append(obj_id_from_usd(root))
        elif root.is_dir():
            for pattern in ("**/*.usd", "*.usd"):
                for p in sorted(root.glob(pattern)):
                    if p.is_file():
                        found.append(obj_id_from_usd(p))

    if candidate_dir:
        cdir = Path(candidate_dir).expanduser()
        if cdir.is_dir():
            for entry in sorted(cdir.iterdir()):
                if entry.is_dir():
                    h5s = list(entry.glob("*_grasp.hdf5"))
                    if h5s:
                        found.append(entry.name)
                    continue
                name = entry.name
                if not name.endswith("_grasp.hdf5"):
                    continue
                if "_yaw" in name:
                    found.append(name.split("_yaw")[0])
                else:
                    found.append(name.replace("_grasp.hdf5", ""))

    # stable unique
    seen: set[str] = set()
    out: list[str] = []
    for oid in found:
        if oid not in seen:
            seen.add(oid)
            out.append(oid)
    return out


def default_candidate_hdf5(candidate_dir: Path, obj_id: str, z_yaw_deg: float) -> Path:
    """Resolve eval_pool / batch pool HDF5 (flat or per-object subdir)."""
    tag = int(round(float(z_yaw_deg))) % 360
    candidates = [
        candidate_dir / f"{obj_id}_yaw{tag:03d}_grasp.hdf5",
        candidate_dir / f"{obj_id}_yaw{tag:03d}_pool_grasp.hdf5",
        candidate_dir / obj_id / f"{obj_id}_yaw{tag:03d}_pool_grasp.hdf5",
        candidate_dir / obj_id / f"{obj_id}_yaw{tag:03d}_grasp.hdf5",
        candidate_dir / f"{obj_id}_grasp.hdf5",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0]


def build_episode_id(
    obj_id: str,
    policy: str,
    trial: int,
    z_yaw_deg: float,
) -> str:
    yaw_tag = int(round(float(z_yaw_deg))) % 360
    return f"{obj_id}_{policy}_yaw{yaw_tag:03d}_t{trial:03d}"


def pick_record_trials(
    trials_per_object: int,
    record_count: int | None,
    rng,
) -> set[int]:
    """Which trial indices (0..N-1) should record video for one object."""
    n = max(0, int(trials_per_object))
    if n == 0:
        return set()
    k = n if record_count is None else min(int(record_count), n)
    if k <= 0:
        return set()
    if k >= n:
        return set(range(n))
    return set(int(x) for x in rng.choice(n, size=k, replace=False))

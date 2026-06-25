#!/usr/bin/env python3
"""
按 merged 中 trusted 成功 pose 数量过滤物体，从 affordance_all(.h5) /
affordance_all_soft.h5 切片写出 train/val HDF5 与 objects_train_val_split.json。

训练（不改 train_v6）::

    python3 tools/filter_affordance_split_by_pose_count.py \\
        --dataset-dir output/affordance_no_rot_executed \\
        --min-trusted 10

    python -m model.train_v6 \\
        --dataset_dir output/affordance_no_rot_executed \\
        --save_dir output/affordance_no_rot_executed/checkpoints_v6_min10 \\
        --epochs 300 --batch_size 64 --num_workers 0

affordance_all* 不会被修改；仅覆盖（或写到 --output-dir）：
  affordance_train.h5, affordance_val.h5,
  affordance_train_soft.h5, affordance_val_soft.h5,
  objects_train_val_split.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import h5py
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MERGED_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged")
sys.path.insert(0, PROJ)

from tools.soft_affordance_gt import EXTRA_DATA_KEYS, decode_obj_ids  # noqa: E402


def is_trusted_grasp(g: h5py.Group) -> bool:
    """Same rules as prepare_affordance_executed.is_trusted_grasp."""
    if "gripper_tips_loc" not in g:
        return False
    if bool(g.attrs.get("gripper_tips_trusted", False)):
        return True
    if str(g.attrs.get("gripper_tips_source", "")) == "legacy_post_lift":
        return False
    return str(g.attrs.get("gripper_tips_snapshot", "at_close")) == "at_close"


def count_trusted_grasps(merged_path: str) -> int:
    with h5py.File(merged_path, "r") as f:
        if "successful_grasps" not in f:
            return 0
        grp = f["successful_grasps"]
        return sum(1 for key in grp.keys() if is_trusted_grasp(grp[key]))


def split_objects(
    obj_ids: list[str],
    train_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Same shuffle split as prepare_affordance_executed.split_objects."""
    ids = sorted(obj_ids)
    if len(ids) == 0:
        return [], []
    if len(ids) == 1:
        return ids, ids
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * (1.0 - train_ratio))))
    n_val = min(n_val, len(ids) - 1)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids

SPLIT_KEYS = ("points", "labels", "obj_ids")
SOFT_KEYS = ("soft_labels", "soft_sigma")


def _load_all_h5(path: str) -> dict:
    with h5py.File(path, "r") as f:
        data: dict[str, np.ndarray] = {
            "points": f["data/points"][:],
            "labels": f["data/labels"][:],
            "obj_ids": f["data/obj_ids"][:],
        }
        for key in SOFT_KEYS:
            p = f"data/{key}"
            if p in f:
                data[key] = f[p][:]
        for key in EXTRA_DATA_KEYS:
            p = f"data/{key}"
            if p in f:
                data[key] = f[p][:]
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
    return {"data": data, "meta": meta}


def _subset_by_obj_ids(data: dict, keep: set[str]) -> dict:
    ids = decode_obj_ids(data["obj_ids"])
    idx = [i for i, oid in enumerate(ids) if oid in keep]
    if not idx:
        raise ValueError("no samples left after filter")
    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        out[key] = arr[idx]
    return out


def _write_h5(path: str, data: dict, meta: dict, *, is_soft: bool) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = int(data["points"].shape[0])
    with h5py.File(path, "w") as f:
        mg = f.create_group("metadata")
        mg.attrs["num_samples"] = n
        mg.attrs["num_points"] = int(data["points"].shape[1])
        mg.attrs["filtered_at"] = datetime.now(timezone.utc).isoformat()
        mg.attrs["is_soft"] = is_soft
        for k, v in meta.items():
            if k in ("num_samples", "num_points"):
                continue
            try:
                mg.attrs[k] = v
            except TypeError:
                mg.attrs[k] = str(v)
        grp = f.create_group("data")
        for key, arr in data.items():
            if key == "obj_ids":
                grp.create_dataset(key, data=arr)
            else:
                grp.create_dataset(
                    key, data=arr, compression="gzip", compression_opts=4,
                )
    return n


def _load_original_split(dataset_dir: str) -> tuple[set[str], set[str]] | None:
    path = os.path.join(dataset_dir, "objects_train_val_split.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    train = {str(x) for x in doc.get("train", [])}
    val = {str(x) for x in doc.get("val", [])}
    if train & val:
        raise ValueError(f"original split has overlap: {train & val}")
    return train, val


def _assign_splits(
    eligible: list[str],
    *,
    mode: str,
    train_ratio: float,
    split_seed: int,
    original: tuple[set[str], set[str]] | None,
) -> tuple[list[str], list[str]]:
    eligible_set = set(eligible)
    if mode == "rebalance":
        return split_objects(sorted(eligible_set), train_ratio, split_seed)

    if original is None:
        print("  ⚠️  No objects_train_val_split.json; using rebalance", file=sys.stderr)
        return split_objects(sorted(eligible_set), train_ratio, split_seed)

    orig_train, orig_val = original
    if mode == "fixed-val":
        train_ids = sorted(eligible_set & orig_train)
        val_ids = sorted(eligible_set & orig_val)
    elif mode == "preserve":
        # Same as fixed-val: keep original membership for objects that pass threshold.
        train_ids = sorted(eligible_set & orig_train)
        val_ids = sorted(eligible_set & orig_val)
    else:
        raise ValueError(f"unknown split mode: {mode}")

    if not train_ids:
        raise ValueError("no train objects after filter (lower --min-trusted or check split)")
    if not val_ids:
        raise ValueError("no val objects after filter (lower --min-trusted or check split)")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"train/val overlap after filter: {overlap}")
    return train_ids, val_ids


def _count_poses(merged_dir: str, obj_ids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for oid in obj_ids:
        path = os.path.join(merged_dir, f"{oid}_robot_gt_merged.hdf5")
        if not os.path.isfile(path):
            out[oid] = 0
            continue
        out[oid] = count_trusted_grasps(path)
    return out


def _backup_paths(dataset_dir: str, names: tuple[str, ...]) -> str | None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(dataset_dir, f"_backup_split_{stamp}")
    os.makedirs(bak, exist_ok=True)
    copied = False
    for name in names:
        src = os.path.join(dataset_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(bak, name))
            copied = True
    return bak if copied else None


def run(args: argparse.Namespace) -> dict:
    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.output_dir or dataset_dir)
    merged_dir = os.path.abspath(args.merged_dir)

    all_bin = os.path.join(dataset_dir, "affordance_all.h5")
    all_soft = os.path.join(dataset_dir, "affordance_all_soft.h5")
    if not os.path.isfile(all_soft):
        raise FileNotFoundError(f"missing {all_soft}")
    if not os.path.isfile(all_bin):
        raise FileNotFoundError(f"missing {all_bin}")

    soft_pack = _load_all_h5(all_soft)
    bin_pack = _load_all_h5(all_bin)
    soft_ids = decode_obj_ids(soft_pack["data"]["obj_ids"])
    bin_ids = decode_obj_ids(bin_pack["data"]["obj_ids"])
    if set(soft_ids) != set(bin_ids):
        only_soft = set(soft_ids) - set(bin_ids)
        only_bin = set(bin_ids) - set(soft_ids)
        raise ValueError(
            f"obj_id mismatch between all_soft and all_bin: "
            f"only_soft={only_soft} only_bin={only_bin}"
        )

    pose_counts = _count_poses(merged_dir, soft_ids)
    eligible = sorted(oid for oid in soft_ids if pose_counts[oid] >= args.min_trusted)
    excluded = sorted(oid for oid in soft_ids if oid not in eligible)

    original = _load_original_split(dataset_dir)
    train_ids, val_ids = _assign_splits(
        eligible,
        mode=args.split_mode,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        original=original,
    )

    train_set, val_set = set(train_ids), set(val_ids)
    soft_train = _subset_by_obj_ids(soft_pack["data"], train_set)
    soft_val = _subset_by_obj_ids(soft_pack["data"], val_set)
    bin_train = _subset_by_obj_ids(bin_pack["data"], train_set)
    bin_val = _subset_by_obj_ids(bin_pack["data"], val_set)

    if args.backup and out_dir == dataset_dir:
        bak = _backup_paths(
            dataset_dir,
            (
                "affordance_train.h5",
                "affordance_val.h5",
                "affordance_train_soft.h5",
                "affordance_val_soft.h5",
                "objects_train_val_split.json",
            ),
        )
        if bak:
            print(f"  Backup → {bak}")

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "train_soft": os.path.join(out_dir, "affordance_train_soft.h5"),
        "val_soft": os.path.join(out_dir, "affordance_val_soft.h5"),
        "train_bin": os.path.join(out_dir, "affordance_train.h5"),
        "val_bin": os.path.join(out_dir, "affordance_val.h5"),
        "split": os.path.join(out_dir, "objects_train_val_split.json"),
        "stats_csv": os.path.join(out_dir, "filter_split_pose_stats.csv"),
    }

    meta_extra = {
        "filter_min_trusted_grasps": args.min_trusted,
        "filter_merged_dir": merged_dir,
        "filter_source_all_soft": all_soft,
        "filter_split_mode": args.split_mode,
    }
    soft_meta = {**soft_pack["meta"], **meta_extra}
    bin_meta = {**bin_pack["meta"], **meta_extra}

    n_tr_soft = _write_h5(paths["train_soft"], soft_train, soft_meta, is_soft=True)
    n_va_soft = _write_h5(paths["val_soft"], soft_val, soft_meta, is_soft=True)
    n_tr_bin = _write_h5(paths["train_bin"], bin_train, bin_meta, is_soft=False)
    n_va_bin = _write_h5(paths["val_bin"], bin_val, bin_meta, is_soft=False)

    split_doc = {
        "train": train_ids,
        "val": val_ids,
        "filter": {
            "min_trusted_grasps": args.min_trusted,
            "merged_dir": merged_dir,
            "split_mode": args.split_mode,
            "train_ratio": args.train_ratio,
            "split_seed": args.split_seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_eligible": len(eligible),
            "n_excluded_below_threshold": len(excluded),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
        },
        "excluded_below_threshold": [
            {"obj_id": oid, "n_trusted": pose_counts[oid]}
            for oid in excluded
        ],
    }
    with open(paths["split"], "w") as f:
        json.dump(split_doc, f, indent=2)

    with open(paths["stats_csv"], "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "obj_id",
                "n_trusted",
                "in_affordance_all",
                "passes_threshold",
                "split",
            ],
        )
        w.writeheader()
        orig_train, orig_val = original if original else (set(), set())
        for oid in sorted(soft_ids, key=lambda x: (-pose_counts[x], x)):
            if oid in train_set:
                split = "train"
            elif oid in val_set:
                split = "val"
            elif oid in excluded:
                split = "excluded_threshold"
            else:
                split = "excluded_other"
            w.writerow({
                "obj_id": oid,
                "n_trusted": pose_counts[oid],
                "in_affordance_all": True,
                "passes_threshold": oid in eligible,
                "split": split,
            })

    report = {
        "dataset_dir": dataset_dir,
        "output_dir": out_dir,
        "min_trusted": args.min_trusted,
        "n_in_all": len(soft_ids),
        "n_eligible": len(eligible),
        "n_excluded": len(excluded),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_soft_samples": n_tr_soft,
        "val_soft_samples": n_va_soft,
        "paths": paths,
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Filter affordance train/val split by trusted grasp pose count",
    )
    p.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(PROJ, "output", "affordance_no_rot_executed"),
        help="Directory with affordance_all.h5 and affordance_all_soft.h5",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write split HDF5 here (default: same as --dataset-dir)",
    )
    p.add_argument(
        "--merged-dir",
        type=str,
        default=DEFAULT_MERGED_DIR,
        help="merged/*_robot_gt_merged.hdf5 for pose counts",
    )
    p.add_argument(
        "--min-trusted",
        type=int,
        required=True,
        help="Keep objects with >= this many trusted successful grasps in merged GT",
    )
    p.add_argument(
        "--split-mode",
        choices=("fixed-val", "preserve", "rebalance"),
        default="fixed-val",
        help=(
            "fixed-val/preserve: keep original train/val membership for eligible objects; "
            "rebalance: reshuffle all eligible objects"
        ),
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing train/val/split in dataset-dir before overwrite",
    )
    args = p.parse_args()
    if args.min_trusted < 1:
        p.error("--min-trusted must be >= 1")

    print("=" * 60)
    print("Filter affordance split by trusted pose count")
    print(f"  dataset_dir:  {args.dataset_dir}")
    print(f"  merged_dir:   {args.merged_dir}")
    print(f"  min_trusted:  {args.min_trusted}")
    print(f"  split_mode:   {args.split_mode}")
    print("=" * 60)

    report = run(args)
    print(f"\n  Eligible: {report['n_eligible']}/{report['n_in_all']} objects")
    print(f"  Train:    {report['n_train']} objects ({report['train_soft_samples']} samples)")
    print(f"  Val:      {report['n_val']} objects ({report['val_soft_samples']} samples)")
    print(f"  Excluded: {report['n_excluded']} (below threshold)")
    print(f"\n  Wrote:")
    for k, path in report["paths"].items():
        print(f"    {k}: {path}")
    print("\n  Next: python -m model.train_v6 --dataset_dir", report["output_dir"])


if __name__ == "__main__":
    main()

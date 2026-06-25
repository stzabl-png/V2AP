#!/usr/bin/env python3
"""
Merge affordance_train.h5 + affordance_val.h5 → affordance_all.h5 (and soft variants).

Does not modify the source train/val files. Creates new combined files only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import h5py
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from tools.soft_affordance_gt import EXTRA_DATA_KEYS, decode_obj_ids, write_soft_h5


def _load_h5(path: str) -> dict:
    with h5py.File(path, "r") as f:
        data = {
            "points": f["data/points"][:],
            "labels": f["data/labels"][:],
            "obj_ids": f["data/obj_ids"][:],
        }
        if "data/soft_labels" in f:
            data["soft_labels"] = f["data/soft_labels"][:]
            data["soft_sigma"] = f["data/soft_sigma"][:]
        for key in EXTRA_DATA_KEYS:
            p = f"data/{key}"
            if p in f:
                data[key] = f[p][:]
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
    return {"path": path, "data": data, "meta": meta}


def _concat_parts(parts: list[dict], *, sort_by_obj_id: bool) -> dict:
    if not parts:
        raise ValueError("no HDF5 parts to merge")

    n_total = sum(p["data"]["points"].shape[0] for p in parts)
    n_pts = parts[0]["data"]["points"].shape[1]
    for p in parts:
        if p["data"]["points"].shape[1] != n_pts:
            raise ValueError("num_points mismatch between splits")

    order = []
    for pi, p in enumerate(parts):
        ids = decode_obj_ids(p["data"]["obj_ids"])
        for j, oid in enumerate(ids):
            if oid in {o for _, _, o in order}:
                raise ValueError(f"duplicate obj_id {oid} in train and val")
            order.append((pi, j, oid))

    if sort_by_obj_id:
        order.sort(key=lambda x: x[2])

    def gather(key: str) -> np.ndarray:
        chunks = [parts[pi]["data"][key][j : j + 1] for pi, j, _ in order]
        return np.concatenate(chunks, axis=0)

    out: dict = {
        "points": gather("points"),
        "labels": gather("labels"),
        "obj_ids": np.array([oid for _, _, oid in order], dtype="S32"),
    }
    for key in EXTRA_DATA_KEYS:
        if key in parts[0]["data"]:
            out[key] = gather(key)
    if "soft_labels" in parts[0]["data"]:
        out["soft_labels"] = gather("soft_labels")
        out["soft_sigma"] = gather("soft_sigma")
    return out


def _write_binary_h5(path: str, data: dict, meta: dict, *, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists (use --overwrite)")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        mg = f.create_group("metadata")
        mg.attrs["merged_at"] = datetime.now().isoformat(timespec="seconds")
        mg.attrs["num_samples"] = int(data["points"].shape[0])
        mg.attrs["num_points"] = int(data["points"].shape[1])
        for k, v in meta.items():
            try:
                mg.attrs[k] = v
            except TypeError:
                mg.attrs[k] = str(v)
        grp = f.create_group("data")
        grp.create_dataset("points", data=data["points"], compression="gzip", compression_opts=4)
        grp.create_dataset("normals", data=data["normals"], compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=data["labels"], compression="gzip", compression_opts=4)
        grp.create_dataset(
            "human_priors", data=data["human_priors"], compression="gzip", compression_opts=4,
        )
        grp.create_dataset(
            "force_centers", data=data["force_centers"], compression="gzip", compression_opts=4,
        )
        grp.create_dataset("obj_ids", data=data["obj_ids"])
        grp.create_dataset("categories", data=data["categories"])
        grp.create_dataset("intents", data=data["intents"])


def merge_dataset_dir(
    dataset_dir: str,
    *,
    overwrite: bool = False,
    sort_by_obj_id: bool = True,
    heatmap_sigma_ratio: float | None = None,
    label_threshold: float = 0.5,
) -> dict:
    dataset_dir = os.path.abspath(dataset_dir)
    train_p = os.path.join(dataset_dir, "affordance_train.h5")
    val_p = os.path.join(dataset_dir, "affordance_val.h5")
    if not os.path.isfile(train_p) or not os.path.isfile(val_p):
        raise FileNotFoundError(f"need {train_p} and {val_p}")

    parts = [_load_h5(train_p), _load_h5(val_p)]
    merged = _concat_parts(parts, sort_by_obj_id=sort_by_obj_id)
    meta = parts[0]["meta"]

    all_bin = os.path.join(dataset_dir, "affordance_all.h5")
    _write_binary_h5(all_bin, merged, meta, overwrite=overwrite)
    print(f"Wrote {all_bin}: {merged['points'].shape[0]} samples")

    extra = {k: merged[k] for k in EXTRA_DATA_KEYS if k in merged}
    all_soft = os.path.join(dataset_dir, "affordance_all_soft.h5")

    train_soft = os.path.join(dataset_dir, "affordance_train_soft.h5")
    val_soft = os.path.join(dataset_dir, "affordance_val_soft.h5")
    if os.path.isfile(train_soft) and os.path.isfile(val_soft):
        soft_parts = [_load_h5(train_soft), _load_h5(val_soft)]
        soft_merged = _concat_parts(soft_parts, sort_by_obj_id=sort_by_obj_id)
        if os.path.exists(all_soft) and not overwrite:
            raise FileExistsError(f"{all_soft} exists (use --overwrite)")
        with h5py.File(all_soft, "w") as f:
            mg = f.create_group("metadata")
            mg.attrs["merged_at"] = datetime.now().isoformat(timespec="seconds")
            mg.attrs["merged_from"] = "affordance_train_soft.h5 + affordance_val_soft.h5"
            mg.attrs["num_samples"] = int(soft_merged["points"].shape[0])
            grp = f.create_group("data")
            for key, arr in soft_merged.items():
                if key == "obj_ids":
                    grp.create_dataset(key, data=arr)
                else:
                    grp.create_dataset(
                        key, data=arr, compression="gzip", compression_opts=4,
                    )
        print(f"Wrote {all_soft}: {soft_merged['points'].shape[0]} samples (from existing soft splits)")
    else:
        info = write_soft_h5(
            all_soft,
            merged["points"],
            merged["labels"],
            merged["obj_ids"],
            extra,
            heatmap_sigma_ratio=heatmap_sigma_ratio or 0.03,
            label_threshold=label_threshold,
            source_h5=all_bin,
            src_meta=meta,
            overwrite=overwrite,
        )
        print(
            f"Wrote {all_soft}: {info['num_samples']} samples "
            f"(σ∈[{info['sigma_m_min']:.4f}, {info['sigma_m_max']:.4f}] m)"
        )

    split_path = os.path.join(dataset_dir, "objects_train_val_split.json")
    report = {
        "dataset_dir": dataset_dir,
        "affordance_all_h5": all_bin,
        "affordance_all_soft_h5": all_soft,
        "num_samples": int(merged["points"].shape[0]),
        "objects": decode_obj_ids(merged["obj_ids"]),
        "split_json": split_path if os.path.isfile(split_path) else None,
        "note": "Source train/val HDF5 files were not modified.",
    }
    report_path = os.path.join(dataset_dir, "merge_all_meta.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Meta: {report_path}")
    return report


def main():
    p = argparse.ArgumentParser(description="Merge train+val affordance HDF5 into affordance_all*")
    p.add_argument(
        "--dataset-dir",
        default=os.path.join(PROJ, "output", "affordance_no_rot_executed"),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-sort", action="store_true", help="Keep train then val order (default: sort obj_id)")
    p.add_argument("--heatmap-sigma-ratio", type=float, default=0.03)
    args = p.parse_args()
    merge_dataset_dir(
        args.dataset_dir,
        overwrite=args.overwrite,
        sort_by_obj_id=not args.no_sort,
        heatmap_sigma_ratio=args.heatmap_sigma_ratio,
    )


if __name__ == "__main__":
    main()

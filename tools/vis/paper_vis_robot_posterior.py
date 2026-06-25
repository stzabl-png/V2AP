#!/usr/bin/env python3
"""
Export robot-posterior soft affordance GT from affordance_no_rot_executed into paper_vis.

Reads ``data/soft_labels`` (+ points/normals) from ``affordance_all_soft.h5`` and writes::

    output/paper_vis/affordance/robot_posterior/
      raw/{obj_id}.npz
      normalized/{obj_id}.npz
      vis/{obj_id}.png

Uses the same ``jet`` colormap and per-object min-max normalization as
``paper_vis_batch`` / ``inference_v6``.

Objects missing from the HDF5 can be prepared on the fly with ``--prepare-missing``.

Usage::

    python tools/paper_vis_robot_posterior.py

    python tools/paper_vis_robot_posterior.py --prepare-missing
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJ / "evaluation" / "configs" / "eval_objects_merged_success_ge30.csv"
DEFAULT_OUT = PROJ / "output" / "paper_vis"
DEFAULT_AFFORDANCE_DIR = PROJ / "output" / "affordance_no_rot_executed"
DEFAULT_SOFT_H5 = DEFAULT_AFFORDANCE_DIR / "affordance_all_soft.h5"

sys.path.insert(0, str(PROJ))

from model.inference_v6 import compose_png_grid, normalize_affordance_pred  # noqa: E402
from tools.affordance_pointcloud_vis import save_scalar_pointcloud_png  # noqa: E402
from tools.paper_vis_batch import load_obj_ids_from_csv  # noqa: E402


def build_obj_index(h5_path: Path) -> dict[str, int]:
    with h5py.File(h5_path, "r") as f:
        ids = f["data/obj_ids"][:]
    return {
        (x.decode() if isinstance(x, bytes) else str(x)): i
        for i, x in enumerate(ids)
    }


def load_soft_sample(h5_path: Path, index: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        return {
            "points": f["data/points"][index].astype(np.float32),
            "normals": f["data/normals"][index].astype(np.float32),
            "soft_labels": f["data/soft_labels"][index].astype(np.float32),
            "labels": f["data/labels"][index].astype(np.float32)
            if "data/labels" in f
            else None,
        }


def prepare_missing_object(obj_id: str, tmp_root: Path) -> dict:
    """Run prepare_affordance_executed for one object; return soft sample arrays."""
    obj_dir = tmp_root / obj_id
    obj_dir.mkdir(parents=True, exist_ok=True)
    script = PROJ / "tools" / "prepare_affordance_executed.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--obj", obj_id, "--outdir", str(obj_dir)],
        cwd=str(PROJ),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"prepare_affordance_executed failed for {obj_id} (exit {proc.returncode})")
    h5 = obj_dir / "affordance_all_soft.h5"
    if not h5.is_file():
        raise FileNotFoundError(f"prepare did not write {h5}")
    return load_soft_sample(h5, 0)


def export_one(
    obj_id: str,
    sample: dict,
    *,
    out_raw: Path,
    out_norm: Path,
    out_vis: Path,
    source_h5: str,
) -> dict:
    points = sample["points"]
    normals = sample["normals"]
    soft = np.asarray(sample["soft_labels"], dtype=np.float32).reshape(-1)
    soft_norm, stats = normalize_affordance_pred(soft)

    base = dict(
        points=points,
        normals=normals,
        obj_id=obj_id,
        source_h5=source_h5,
        supervision="robot_posterior_soft_gt",
    )
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_norm.parent.mkdir(parents=True, exist_ok=True)
    out_vis.parent.mkdir(parents=True, exist_ok=True)

    np.savez(out_raw, soft_labels=soft, **base)
    np.savez(
        out_norm,
        soft_labels=soft_norm,
        soft_labels_raw=soft,
        **base,
        **{f"soft_{k}": v for k, v in stats.items()},
    )
    save_scalar_pointcloud_png(
        out_vis,
        points,
        soft_norm,
        f"{obj_id}  robot posterior soft GT (normalized)",
        vmax=1.0,
    )
    return {
        "obj_id": obj_id,
        "robot_posterior_raw_npz": str(out_raw),
        "robot_posterior_norm_npz": str(out_norm),
        "robot_posterior_vis": str(out_vis),
        "soft_max": float(soft.max()),
        "soft_mean": float(soft.mean()),
        "soft_norm_stats": stats,
        "source_h5": source_h5,
    }


def merge_summary(out_root: Path, rows: list[dict], skipped: list[dict]) -> None:
    summary_path = out_root / "summary.json"
    doc: dict = {}
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            doc = json.load(f)
    by_id = {r["obj_id"]: r for r in doc.get("objects", []) if "obj_id" in r}
    for row in rows:
        oid = row["obj_id"]
        by_id[oid] = {**by_id.get(oid, {}), **row}
    doc["robot_posterior_h5"] = str(DEFAULT_SOFT_H5)
    doc["robot_posterior_exported"] = len(rows)
    doc["robot_posterior_skipped"] = skipped
    doc["objects"] = [by_id[k] for k in sorted(by_id.keys())]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Export robot posterior soft GT to paper_vis")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--affordance-h5", type=Path, default=DEFAULT_SOFT_H5)
    p.add_argument("--obj", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--prepare-missing",
        action="store_true",
        help="Run prepare_affordance_executed for objects not in affordance_all_soft.h5",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_root = args.output_dir.expanduser().resolve()
    rp_root = out_root / "affordance" / "robot_posterior"
    raw_dir = rp_root / "raw"
    norm_dir = rp_root / "normalized"
    vis_dir = rp_root / "vis"

    if args.obj:
        obj_ids = list(dict.fromkeys(args.obj))
    else:
        obj_ids = load_obj_ids_from_csv(args.csv.expanduser().resolve())
    if args.limit > 0:
        obj_ids = obj_ids[: args.limit]

    h5_path = args.affordance_h5.expanduser().resolve()
    if not h5_path.is_file():
        raise SystemExit(f"affordance soft H5 not found: {h5_path}")

    index = build_obj_index(h5_path)
    missing = [o for o in obj_ids if o not in index]
    tmp_root = DEFAULT_AFFORDANCE_DIR / "_paper_vis_prepare_tmp"
    prepared_missing: dict[str, dict] = {}
    if missing and args.prepare_missing:
        print(f"Preparing {len(missing)} missing object(s): {missing}")
        for oid in list(missing):
            try:
                prepared_missing[oid] = prepare_missing_object(oid, tmp_root)
            except Exception as exc:
                print(f"  prepare failed {oid}: {exc}")
        missing = [o for o in missing if o not in prepared_missing]

    print("=" * 72)
    print("paper_vis_robot_posterior")
    print(f"  objects:  {len(obj_ids)}")
    print(f"  h5:       {h5_path}")
    print(f"  output:   {rp_root}")
    print(f"  missing:  {len([o for o in obj_ids if o not in index])}")
    print("=" * 72)

    rows: list[dict] = []
    skipped: list[dict] = []

    for i, oid in enumerate(obj_ids, 1):
        if oid not in index and oid not in prepared_missing:
            print(f"[{i}/{len(obj_ids)}] skip {oid}: not in affordance HDF5")
            skipped.append({"obj_id": oid, "reason": "not_in_h5"})
            continue

        out_vis = vis_dir / f"{oid}.png"
        if not args.overwrite and out_vis.is_file():
            print(f"[{i}/{len(obj_ids)}] skip existing {oid}")
            continue

        print(f"[{i}/{len(obj_ids)}] export {oid}")
        if oid in prepared_missing:
            sample = prepared_missing[oid]
            source = str(tmp_root / oid / "affordance_all_soft.h5")
        else:
            sample = load_soft_sample(h5_path, index[oid])
            source = str(h5_path)
        row = export_one(
            oid,
            sample,
            out_raw=raw_dir / f"{oid}.npz",
            out_norm=norm_dir / f"{oid}.npz",
            out_vis=out_vis,
            source_h5=source,
        )
        rows.append(row)
        print(f"  soft max={row['soft_max']:.3f} mean={row['soft_mean']:.4f}")

    vis_pngs = sorted(vis_dir.glob("*.png"))
    if len(vis_pngs) > 1:
        compose_png_grid(
            [str(p) for p in vis_pngs],
            str(vis_dir / "overview.png"),
            cols=min(5, len(vis_pngs)),
        )

    merge_summary(out_root, rows, skipped)
    print(f"\nExported {len(rows)} robot posterior soft maps -> {rp_root}")
    if skipped:
        print(f"Skipped {len(skipped)} (see summary.json robot_posterior_skipped)")
    print("Done.")


if __name__ == "__main__":
    main()

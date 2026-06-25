#!/usr/bin/env python3
"""
Batch paper figures: robot posterior affordance + PDM candidates + human prior.

Reads object ids from a CSV (default: eval_objects_merged_success_ge30.csv),
runs affordance v6 + PDM on rotated SAM3D meshes, exports point clouds and PNGs
with a shared ``jet`` colormap (blue=low, red=high).

Output layout::

    output/paper_vis/
      hp/
        {obj_id}.npz              # hp_points, human_prior
        vis/{obj_id}.png
      affordance/
        raw/{obj_id}.npz          # v6 pred raw + points/normals
        normalized/{obj_id}.npz   # min-max pred in [0,1]
        vis/{obj_id}.png          # normalized pred (jet heatmap)
        points_vis/{obj_id}.png   # same points, uniform gray (no heatmap)
        robot_posterior/          # GT soft from affordance_no_rot_executed (see paper_vis_robot_posterior.py)
          raw/ normalized/ vis/
      candidate_poses/
        {obj_id}_grasp.hdf5
        vis/{obj_id}.png          # top-K grippers on normalized affordance cloud
      compare/
        {obj_id}.png              # HP | robot posterior | affordance (tools/paper_vis_triptych.py)
      summary.json

Usage::

    python tools/paper_vis_batch.py

    python tools/paper_vis_batch.py --limit 3 --obj A01001 A02015

    python tools/paper_vis_batch.py \\
        --csv evaluation/configs/eval_objects_merged_success_ge30.csv \\
        --output-dir output/paper_vis \\
        --n-samples 20 --vis-top 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJ / "evaluation" / "configs" / "eval_objects_merged_success_ge30.csv"
DEFAULT_OUT = PROJ / "output" / "paper_vis"
DEFAULT_MESH_ROOT = PROJ / "data_hub" / "meshes" / "SAM3DMesh" / "rotated_mesh"
DEFAULT_PDM_CKPT = PROJ / "output" / "pdm" / "checkpoints_yaw_v6cond" / "best_model.pth"

sys.path.insert(0, str(PROJ))

from evaluation.affordance_ckpt import (  # noqa: E402
    add_affordance_checkpoint_args,
    resolve_affordance_checkpoint,
)
from evaluation.eval_single import infer_rotated_mesh_dataset, resolve_generate_mesh  # noqa: E402
from model.inference_v6 import (  # noqa: E402
    compose_png_grid,
    default_threshold,
    load_model,
    normalize_affordance_pred,
    predict_heatmap_batch,
)
from tools.affordance_pointcloud_vis import (  # noqa: E402
    PLAIN_POINT_RGB,
    save_pointcloud_png,
    save_scalar_pointcloud_png,
)
from model.pdm.model import PDM  # noqa: E402
from model.pdm.sample import write_candidates_hdf5  # noqa: E402
from model.pdm.visualize import make_overview, save_candidate_overlay  # noqa: E402
from tools.glb_to_pdm_grasp import (  # noqa: E402
    build_condition_tensor,
    output_hdf5_name,
    prepare_mesh_item,
    run_pdm_sample,
    scale_report_row,
)
from tools.vis_rotated_mesh_hp import load_mesh_and_hp  # noqa: E402


@dataclass
class PaperVisDirs:
    root: Path
    hp: Path
    hp_vis: Path
    aff_raw: Path
    aff_norm: Path
    aff_vis: Path
    aff_points_vis: Path
    cand: Path
    cand_vis: Path

    @classmethod
    def create(cls, root: Path) -> "PaperVisDirs":
        d = cls(
            root=root,
            hp=root / "hp",
            hp_vis=root / "hp" / "vis",
            aff_raw=root / "affordance" / "raw",
            aff_norm=root / "affordance" / "normalized",
            aff_vis=root / "affordance" / "vis",
            aff_points_vis=root / "affordance" / "points_vis",
            cand=root / "candidate_poses",
            cand_vis=root / "candidate_poses" / "vis",
        )
        for p in (
            d.hp,
            d.hp_vis,
            d.aff_raw,
            d.aff_norm,
            d.aff_vis,
            d.aff_points_vis,
            d.cand,
            d.cand_vis,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return d


def load_obj_ids_from_csv(csv_path: Path) -> list[str]:
    ids: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("enabled", "1")).strip().lower() in ("0", "false", "no"):
                continue
            oid = str(row.get("obj_id", "")).strip()
            if oid:
                ids.append(oid)
    return ids


def save_affordance_npz(
    path: Path,
    *,
    points: np.ndarray,
    normals: np.ndarray,
    pred: np.ndarray,
    obj_id: str,
    mesh_path: str,
    extra: dict | None = None,
) -> None:
    kw = dict(
        points=points,
        normals=normals,
        pred=pred,
        gt=None,
        obj_id=obj_id,
        mesh_path=mesh_path,
    )
    if extra:
        kw.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **kw)


def export_hp(obj_id: str, dataset: str, dirs: PaperVisDirs) -> dict:
    mesh, hp_pc, hp_labels, hp_path, scale_factor, ds = load_mesh_and_hp(
        obj_id, dataset, use_legacy_assets=False,
    )
    if hp_pc is None or hp_labels is None:
        raise RuntimeError(f"human prior missing for {obj_id} (dataset={ds})")

    labels = np.asarray(hp_labels, dtype=np.float32).reshape(-1)
    hp_norm, hp_stats = normalize_affordance_pred(labels)

    npz_path = dirs.hp / f"{obj_id}.npz"
    np.savez(
        npz_path,
        hp_points=np.asarray(hp_pc, dtype=np.float32),
        human_prior=labels,
        human_prior_normalized=hp_norm,
        obj_id=obj_id,
        dataset=ds,
        hp_path=hp_path or "",
        scale_factor=float(scale_factor) if scale_factor is not None else 1.0,
        mesh_path=str(mesh.vertices.shape[0]) if mesh is not None else "",
    )

    png_path = dirs.hp_vis / f"{obj_id}.png"
    save_scalar_pointcloud_png(
        png_path,
        hp_pc,
        hp_norm,
        f"{obj_id}  human prior (normalized)",
        vmax=1.0,
    )
    return {
        "obj_id": obj_id,
        "dataset": ds,
        "hp_npz": str(npz_path),
        "hp_vis": str(png_path),
        "hp_path": hp_path,
        "n_hp_points": int(len(hp_pc)),
        "hp_max": float(labels.max()),
        "hp_mean": float(labels.mean()),
        **{f"hp_norm_{k}": v for k, v in hp_stats.items()},
    }


def process_one_object(
    obj_id: str,
    *,
    dirs: PaperVisDirs,
    aff_model,
    pdm_model,
    pdm_stats: dict,
    device: torch.device,
    mesh_root: Path,
    dataset: str | None,
    aff_thresh: float,
    n_samples: int,
    ddim_steps: int,
    vis_top: int,
    z_yaw_deg: float | None,
    gripper_width: float,
    num_points: int,
    seed: int,
    index: int,
    reject_upward: bool,
    max_approach_z: float,
) -> dict:
    ds = infer_rotated_mesh_dataset(obj_id, dataset)
    mesh_args = SimpleNamespace(
        obj_id=obj_id,
        mesh=None,
        mesh_root=str(mesh_root),
        dataset=ds,
        sam3d_rotated_mesh=False,
    )
    mesh_path, _ = resolve_generate_mesh(mesh_args)

    item = prepare_mesh_item(
        mesh_path,
        obj_id_override=obj_id,
        dataset=ds,
        sam3d_rotated_mesh=True,
        no_pre_rotate_x=True,
        pre_rotate_x_deg=90.0,
        scale_mode="never",
        ignore_scale_json=True,
        target_max_extent=0.30,
        auto_extent_lo=0.08,
        auto_extent_hi=0.38,
        min_scale_factor=0.45,
        no_center=True,
        num_points=num_points,
        seed=seed + index,
        index=index,
    )

    pts = item["points"]
    nrm = item["normals"]
    pred = predict_heatmap_batch(
        aff_model,
        pts[np.newaxis, ...],
        nrm[np.newaxis, ...],
        device,
    )[0].astype(np.float32)
    pred_norm, pred_norm_stats = normalize_affordance_pred(pred)

    scale_extra = scale_report_row(item)
    save_affordance_npz(
        dirs.aff_raw / f"{obj_id}.npz",
        points=pts,
        normals=nrm,
        pred=pred,
        obj_id=obj_id,
        mesh_path=str(mesh_path),
        extra=scale_extra,
    )
    save_affordance_npz(
        dirs.aff_norm / f"{obj_id}.npz",
        points=pts,
        normals=nrm,
        pred=pred_norm,
        obj_id=obj_id,
        mesh_path=str(mesh_path),
        extra={**scale_extra, "pred_raw": pred, **pred_norm_stats},
    )
    aff_png = dirs.aff_vis / f"{obj_id}.png"
    save_scalar_pointcloud_png(aff_png, pts, pred_norm, vmax=1.0)

    aff_points_png = dirs.aff_points_vis / f"{obj_id}.png"
    save_pointcloud_png(aff_points_png, pts, solid_rgb=PLAIN_POINT_RGB)

    condition = build_condition_tensor(pts, nrm, pred)
    poses_np = run_pdm_sample(
        pdm_model,
        pdm_stats,
        condition,
        n_samples=n_samples,
        ddim_steps=ddim_steps,
        z_yaw_deg=z_yaw_deg,
        device=device,
        reject_upward=reject_upward,
        max_approach_z=max_approach_z,
    )

    cand_hdf5 = None
    cand_png = None
    n_candidates = 0
    if poses_np.size > 0:
        n_candidates = int(len(poses_np))
        out_name = output_hdf5_name(obj_id, z_yaw_deg)
        cand_hdf5 = dirs.cand / out_name
        write_candidates_hdf5(
            str(cand_hdf5),
            obj_id,
            poses_np,
            mesh_path=str(mesh_path),
            gripper_width=gripper_width,
            dataset=ds,
        )
        cand_png = dirs.cand_vis / f"{obj_id}.png"
        save_candidate_overlay(
            str(cand_hdf5),
            pts,
            str(cand_png),
            top=min(vis_top, n_candidates),
            affordance=pred_norm,
            affordance_vmax_fixed=1.0,
            title_suffix=f"  n={n_candidates}",
        )

    hp_info = export_hp(obj_id, ds, dirs)

    return {
        "obj_id": obj_id,
        "dataset": ds,
        "mesh_path": str(mesh_path),
        "affordance_threshold": float(aff_thresh),
        "affordance_max": float(pred.max()),
        "affordance_mean": float(pred.mean()),
        "affordance_norm_stats": pred_norm_stats,
        "affordance_raw_npz": str(dirs.aff_raw / f"{obj_id}.npz"),
        "affordance_norm_npz": str(dirs.aff_norm / f"{obj_id}.npz"),
        "affordance_vis": str(aff_png),
        "affordance_points_vis": str(aff_points_png),
        "n_candidates": n_candidates,
        "candidate_hdf5": str(cand_hdf5) if cand_hdf5 else None,
        "candidate_vis": str(cand_png) if cand_png else None,
        **hp_info,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch paper vis: affordance + PDM + human prior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    p.add_argument("--obj", nargs="*", default=None, help="Subset of obj ids (overrides csv order)")
    p.add_argument("--limit", type=int, default=0, help="Process at most N objects (0=all)")
    add_affordance_checkpoint_args(p)
    p.add_argument("--pdm-checkpoint", type=Path, default=DEFAULT_PDM_CKPT)
    p.add_argument("--pose-stats", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--vis-top", type=int, default=20)
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z-yaw-deg", type=float, default=None)
    p.add_argument("--gripper-width", type=float, default=0.06)
    p.add_argument("--reject-upward", action="store_true")
    p.add_argument("--max-approach-z", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="Skip if affordance vis png exists")
    args = p.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if args.obj:
        obj_ids = list(dict.fromkeys(args.obj))
    else:
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found: {csv_path}")
        obj_ids = load_obj_ids_from_csv(csv_path)
    if args.limit > 0:
        obj_ids = obj_ids[: args.limit]
    if not obj_ids:
        raise SystemExit("No objects to process")

    out_root = args.output_dir.expanduser().resolve()
    dirs = PaperVisDirs.create(out_root)

    aff_ckpt = resolve_affordance_checkpoint(
        hp_affordance=bool(args.hp_affordance),
        affordance_checkpoint=args.affordance_checkpoint,
    )
    pdm_ckpt = args.pdm_checkpoint.expanduser().resolve()
    if not pdm_ckpt.is_file():
        raise SystemExit(f"PDM checkpoint not found: {pdm_ckpt}")

    device = torch.device(
        args.device or ("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")),
    )
    dataset_dir = args.dataset_dir or aff_ckpt.parent.parent
    aff_thresh = default_threshold(str(aff_ckpt), str(dataset_dir.expanduser().resolve()))

    aff_model, _ = load_model(str(aff_ckpt), device)
    pdm_model, pdm_ckpt_meta = PDM.load(str(pdm_ckpt), device=device)
    pdm_stats = pdm_ckpt_meta.get("pose_stats")
    if pdm_stats is None:
        if args.pose_stats is None:
            raise SystemExit("PDM checkpoint has no pose_stats; pass --pose-stats")
        pdm_stats = torch.load(
            args.pose_stats.expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )

    print("=" * 72)
    print("paper_vis_batch")
    print(f"  Objects:     {len(obj_ids)}")
    print(f"  CSV:         {csv_path}")
    print(f"  Output:      {out_root}")
    print(f"  Aff ckpt:    {aff_ckpt}")
    print(f"  PDM ckpt:    {pdm_ckpt}")
    print(f"  n_samples:   {args.n_samples}  vis_top={args.vis_top}")
    print(f"  Colormap:    jet (shared)")
    print("=" * 72)

    results: list[dict] = []
    skipped: list[dict] = []
    for i, oid in enumerate(obj_ids):
        if args.skip_existing and (dirs.aff_vis / f"{oid}.png").is_file():
            print(f"[{i+1}/{len(obj_ids)}] skip existing {oid}")
            skipped.append({"obj_id": oid, "reason": "existing"})
            continue
        print(f"\n[{i+1}/{len(obj_ids)}] {oid}")
        try:
            row = process_one_object(
                oid,
                dirs=dirs,
                aff_model=aff_model,
                pdm_model=pdm_model,
                pdm_stats=pdm_stats,
                device=device,
                mesh_root=args.mesh_root.expanduser().resolve(),
                dataset=None,
                aff_thresh=aff_thresh,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                vis_top=args.vis_top,
                z_yaw_deg=args.z_yaw_deg,
                gripper_width=args.gripper_width,
                num_points=args.num_points,
                seed=args.seed,
                index=i,
                reject_upward=args.reject_upward,
                max_approach_z=args.max_approach_z,
            )
            results.append(row)
            print(
                f"  affordance max={row['affordance_max']:.3f}  "
                f"candidates={row['n_candidates']}  hp_n={row['n_hp_points']}",
            )
        except Exception as exc:
            print(f"  ERROR {oid}: {type(exc).__name__}: {exc}")
            skipped.append({"obj_id": oid, "reason": f"{type(exc).__name__}: {exc}"})

    aff_pngs = sorted(dirs.aff_vis.glob("*.png"))
    aff_points_pngs = sorted(dirs.aff_points_vis.glob("*.png"))
    cand_pngs = sorted(dirs.cand_vis.glob("*.png"))
    hp_pngs = sorted(dirs.hp_vis.glob("*.png"))
    if len(aff_pngs) > 1:
        compose_png_grid(
            [str(p) for p in aff_pngs if p.name != "overview.png"],
            str(dirs.aff_vis / "overview.png"),
            cols=min(5, len(aff_pngs)),
        )
    if len(aff_points_pngs) > 1:
        compose_png_grid(
            [str(p) for p in aff_points_pngs if p.name != "overview.png"],
            str(dirs.aff_points_vis / "overview.png"),
            cols=min(5, len(aff_points_pngs)),
        )
    if len(cand_pngs) > 1:
        make_overview(
            [str(p) for p in cand_pngs],
            str(dirs.cand_vis / "overview.png"),
            cols=min(5, len(cand_pngs)),
        )
    if len(hp_pngs) > 1:
        compose_png_grid(
            [str(p) for p in hp_pngs],
            str(dirs.hp_vis / "overview.png"),
            cols=min(5, len(hp_pngs)),
        )

    summary = {
        "csv": str(csv_path),
        "output_dir": str(out_root),
        "affordance_checkpoint": str(aff_ckpt),
        "pdm_checkpoint": str(pdm_ckpt),
        "n_samples": args.n_samples,
        "vis_top": args.vis_top,
        "colormap": "jet",
        "n_ok": len(results),
        "n_skipped": len(skipped),
        "objects": results,
        "skipped": skipped,
    }
    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(f"Done: {len(results)} ok, {len(skipped)} skipped.")


if __name__ == "__main__":
    main()

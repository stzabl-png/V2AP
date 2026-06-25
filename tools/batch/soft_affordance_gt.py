#!/usr/bin/env python3
"""Gaussian soft affordance GT — shared by prepare and merge."""

from __future__ import annotations

import json
import os
from datetime import datetime

import h5py
import numpy as np

EXTRA_DATA_KEYS = ("normals", "human_priors", "force_centers", "categories", "intents")

SOFT_EXPORT_PAIRS = (
    ("affordance_all.h5", "affordance_all_soft.h5"),
    ("affordance_train.h5", "affordance_train_soft.h5"),
    ("affordance_val.h5", "affordance_val_soft.h5"),
)


def object_bbox_diagonal(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return 1.0
    extent = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(extent) + 1e-8)


def sigma_from_scale(object_scale: float, sigma_ratio: float) -> float:
    return float(sigma_ratio * object_scale)


def gaussian_soft_heatmap(
    points: np.ndarray,
    binary_contact: np.ndarray,
    sigma: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    contact = np.asarray(binary_contact, dtype=bool).reshape(-1)
    n = pts.shape[0]
    heatmap = np.zeros(n, dtype=np.float32)
    if not contact.any():
        return heatmap

    pos_pts = pts[contact]
    diff = pts[:, None, :] - pos_pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2).min(axis=1)
    heatmap = np.exp(-d2 / (2.0 * float(sigma) ** 2 + 1e-12)).astype(np.float32)
    heatmap[contact] = 1.0
    return heatmap


def decode_obj_ids(raw) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def compute_soft_labels_for_file(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    heatmap_sigma_ratio: float,
    label_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    soft = np.zeros_like(labels, dtype=np.float32)
    sigmas = np.zeros(n, dtype=np.float32)
    binary = labels > label_threshold

    for i in range(n):
        pts = points[i]
        sigma = sigma_from_scale(object_bbox_diagonal(pts), heatmap_sigma_ratio)
        sigmas[i] = sigma
        soft[i] = gaussian_soft_heatmap(pts, binary[i], sigma)
    return soft, sigmas


def write_soft_h5(
    dst_path: str,
    points: np.ndarray,
    labels: np.ndarray,
    obj_ids: np.ndarray,
    extra_data: dict[str, np.ndarray],
    *,
    heatmap_sigma_ratio: float,
    label_threshold: float = 0.5,
    source_h5: str | None = None,
    src_meta: dict | None = None,
    overwrite: bool = False,
) -> dict:
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"{dst_path} exists (use --overwrite)")

    soft, sigmas = compute_soft_labels_for_file(
        points,
        labels,
        heatmap_sigma_ratio=heatmap_sigma_ratio,
        label_threshold=label_threshold,
    )

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with h5py.File(dst_path, "w") as f:
        meta = f.create_group("metadata")
        if source_h5:
            meta.attrs["source_h5"] = os.path.abspath(source_h5)
        meta.attrs["exported_at"] = datetime.now().isoformat(timespec="seconds")
        meta.attrs["heatmap_sigma_ratio"] = float(heatmap_sigma_ratio)
        meta.attrs["label_threshold"] = float(label_threshold)
        meta.attrs["object_scale"] = "bbox_diagonal(points)"
        meta.attrs["sigma_formula"] = "sigma = heatmap_sigma_ratio * object_scale"
        meta.attrs["soft_formula"] = (
            "soft[i]=exp(-d_i^2/(2*sigma^2)), d_i=min dist to contact; contact points=1.0"
        )
        meta.attrs["num_samples"] = int(points.shape[0])
        meta.attrs["num_points"] = int(points.shape[1])
        if src_meta:
            for k, v in src_meta.items():
                try:
                    meta.attrs[f"source_{k}"] = v
                except TypeError:
                    meta.attrs[f"source_{k}"] = str(v)

        grp = f.create_group("data")
        grp.create_dataset("points", data=points, compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=labels, compression="gzip", compression_opts=4)
        grp.create_dataset("soft_labels", data=soft, compression="gzip", compression_opts=4)
        grp.create_dataset("soft_sigma", data=sigmas, compression="gzip", compression_opts=4)
        grp.create_dataset("obj_ids", data=obj_ids)
        for k, arr in extra_data.items():
            grp.create_dataset(k, data=arr, compression="gzip", compression_opts=4)

    decoded = decode_obj_ids(obj_ids)
    return {
        "source": os.path.basename(source_h5) if source_h5 else "",
        "output": os.path.basename(dst_path),
        "num_samples": int(points.shape[0]),
        "heatmap_sigma_ratio": heatmap_sigma_ratio,
        "sigma_m_min": float(sigmas.min()),
        "sigma_m_max": float(sigmas.max()),
        "sigma_m_mean": float(sigmas.mean()),
        "soft_max_mean": float(soft.max(axis=1).mean()),
        "soft_mean_mean": float(soft.mean(axis=1).mean()),
        "contact_frac_mean": float((labels > label_threshold).mean(axis=1).mean()),
        "objects": sorted(set(decoded)),
    }


def export_soft_from_binary_h5(
    src_path: str,
    dst_path: str,
    *,
    heatmap_sigma_ratio: float,
    label_threshold: float,
    overwrite: bool,
) -> dict:
    with h5py.File(src_path, "r") as f:
        points = f["data/points"][:]
        labels = f["data/labels"][:]
        obj_ids = f["data/obj_ids"][:]
        extra_data = {}
        for key in EXTRA_DATA_KEYS:
            path = f"data/{key}"
            if path in f:
                extra_data[key] = f[path][:]
        src_meta = dict(f["metadata"].attrs) if "metadata" in f else {}

    return write_soft_h5(
        dst_path,
        points,
        labels,
        obj_ids,
        extra_data,
        heatmap_sigma_ratio=heatmap_sigma_ratio,
        label_threshold=label_threshold,
        source_h5=src_path,
        src_meta=src_meta,
        overwrite=overwrite,
    )


def export_soft_dataset_dir(
    dataset_dir: str,
    *,
    heatmap_sigma_ratio: float = 0.03,
    label_threshold: float = 0.5,
    overwrite: bool = False,
) -> dict:
    """Refresh *_soft.h5 from existing binary affordance_*.h5 in dataset_dir."""
    dataset_dir = os.path.abspath(dataset_dir)
    report = {
        "dataset_dir": dataset_dir,
        "heatmap_sigma_ratio": heatmap_sigma_ratio,
        "label_threshold": label_threshold,
        "note": "export-soft-only: canonical H5 points, no augmentation",
        "splits": [],
    }

    for src_name, dst_name in SOFT_EXPORT_PAIRS:
        src = os.path.join(dataset_dir, src_name)
        dst = os.path.join(dataset_dir, dst_name)
        if not os.path.isfile(src):
            continue
        info = export_soft_from_binary_h5(
            src,
            dst,
            heatmap_sigma_ratio=heatmap_sigma_ratio,
            label_threshold=label_threshold,
            overwrite=overwrite,
        )
        report["splits"].append(info)
        print(
            f"Wrote {dst_name}: {info['num_samples']} samples, "
            f"σ∈[{info['sigma_m_min']:.4f}, {info['sigma_m_max']:.4f}] m"
        )

    meta_path = os.path.join(dataset_dir, "soft_gt_export_meta.json")
    with open(meta_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Meta: {meta_path}")
    return report

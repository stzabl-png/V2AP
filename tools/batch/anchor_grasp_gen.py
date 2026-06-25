#!/usr/bin/env python3
"""
Anchored candidate generation (Strategy A): perturb successful merged grasps.

Used by random_grasp_sampler --sampling-mode anchored and batch_gen_candidates_pool.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

# Reuse sampler geometry / scoring (import lazily in callers to avoid cycles)
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLING_METHOD_ANCHORED = "anchored_contact_v2"

GRIPPER_WIDTH_PAD = 0.005


@dataclass
class AnchorRecord:
    key: str
    name: str
    score: float
    grasp_point: np.ndarray
    rotation: np.ndarray
    gripper_width: float
    tip_l: np.ndarray | None = None
    tip_r: np.ndarray | None = None
    finger_width_actual: float = 0.0
    mesh_prerotation: dict | None = None


def is_trusted_grasp(g: h5py.Group) -> bool:
    if "gripper_tips_loc" not in g:
        return False
    if bool(g.attrs.get("gripper_tips_trusted", False)):
        return True
    if str(g.attrs.get("gripper_tips_source", "")) == "legacy_post_lift":
        return False
    return str(g.attrs.get("gripper_tips_snapshot", "at_close")) == "at_close"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _orthogonalize_finger(finger_dir: np.ndarray, approach: np.ndarray) -> np.ndarray:
    f = np.asarray(finger_dir, dtype=np.float64).reshape(3)
    a = _unit(approach)
    f = f - np.dot(f, a) * a
    n = np.linalg.norm(f)
    if n < 1e-9:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, a)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        f = np.cross(a, ref)
        n = np.linalg.norm(f)
    return (f / n).astype(np.float32)


def _load_one_anchor(g: h5py.Group, key: str, file_root: h5py.File | None = None) -> AnchorRecord | None:
    if "grasp_point" not in g or "rotation" not in g:
        return None
    R = np.asarray(g["rotation"][:], dtype=np.float64).reshape(3, 3)
    gp = np.asarray(g["grasp_point"][:], dtype=np.float64).reshape(3)
    gw = float(g.attrs.get("gripper_width", 0.04))
    tip_l = tip_r = None
    fwa = 0.0
    if "gripper_tips_loc" in g:
        tips = np.asarray(g["gripper_tips_loc"][:], dtype=np.float64)
        tip_l, tip_r = tips[0], tips[1]
        fwa = float(g.attrs.get("finger_width_actual", 0.0))
    from mesh_utils import read_mesh_prerotation_hdf5_pose

    pr = read_mesh_prerotation_hdf5_pose(g, file_root)
    return AnchorRecord(
        key=str(key),
        name=str(g.attrs.get("name", key)),
        score=float(g.attrs.get("score", 0.0)),
        grasp_point=gp,
        rotation=R.astype(np.float32),
        gripper_width=gw,
        tip_l=tip_l,
        tip_r=tip_r,
        finger_width_actual=fwa,
        mesh_prerotation=pr,
    )


def _load_successful_rows(merged_path: str, *, trusted_only: bool) -> list[AnchorRecord]:
    rows: list[AnchorRecord] = []
    with h5py.File(merged_path, "r") as f:
        if "successful_grasps" not in f:
            return rows
        for key in sorted(f["successful_grasps"].keys()):
            g = f["successful_grasps"][key]
            if trusted_only and not is_trusted_grasp(g):
                continue
            rec = _load_one_anchor(g, key, f)
            if rec is not None:
                rows.append(rec)
    return rows


def load_anchors_from_merged(
    merged_path: str,
    *,
    filter_trusted: bool = True,
    allow_fallback_all: bool = True,
) -> tuple[list[AnchorRecord], dict[str, Any]]:
    """Load anchors; fallback to all successful if trusted pool is empty."""
    meta: dict[str, Any] = {
        "merged_path": os.path.abspath(merged_path),
        "anchor_filter_requested": "trusted" if filter_trusted else "all",
    }
    if not os.path.isfile(merged_path):
        meta["anchor_filter_used"] = "none"
        meta["error"] = "missing_merged_file"
        return [], meta

    with h5py.File(merged_path, "r") as f:
        n_total = 0
        if "successful_grasps" in f:
            n_total = len(f["successful_grasps"].keys())

    trusted = _load_successful_rows(merged_path, trusted_only=True) if filter_trusted else []
    used_filter = "trusted"
    pool = trusted
    if not pool and allow_fallback_all and n_total > 0:
        pool = _load_successful_rows(merged_path, trusted_only=False)
        used_filter = "all_successful_fallback"
    elif not filter_trusted:
        pool = _load_successful_rows(merged_path, trusted_only=False)
        used_filter = "all"

    anchors = sorted(pool, key=lambda r: -r.score)

    meta.update({
        "anchor_filter_used": used_filter,
        "n_successful_total": n_total,
        "n_trusted_loaded": len(trusted),
        "n_anchors": len(anchors),
        "anchor_dedup": False,
    })
    return anchors, meta


def compute_max_retry_per_slot(target_n: int, n_schedule_slots: int) -> int:
    """
    Total perturbation attempt budget = 100×target, split evenly across schedule slots.
    With len(schedule)==target this is 100 retries per slot.
    """
    budget = max(int(target_n) * 100, 1)
    return max(budget // max(int(n_schedule_slots), 1), 1)


def build_anchor_schedule(
    n_anchors: int,
    target: int,
    rng: np.random.Generator,
) -> list[int]:
    """Return length-`target` list of anchor indices (equal split + random remainder)."""
    if n_anchors < 1 or target < 1:
        return []
    K, T = n_anchors, target
    if T <= K:
        picks = rng.choice(K, size=T, replace=False)
        return picks.tolist()
    base = T // K
    rem = T % K
    schedule: list[int] = []
    for i in range(K):
        schedule.extend([i] * base)
    if rem > 0:
        extra = rng.choice(K, size=rem, replace=False)
        for idx in extra:
            schedule.append(int(idx))
    rng.shuffle(schedule)
    return schedule


def _raycast_contacts(
    mesh_rc,
    grasp_center: np.ndarray,
    finger_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    fd = _unit(finger_dir).astype(np.float32)
    center = np.asarray(grasp_center, dtype=np.float64)
    hits_pos, _, _ = mesh_rc.ray.intersects_location([center], [fd])
    hits_neg, _, _ = mesh_rc.ray.intersects_location([center], [-fd])
    if len(hits_pos) == 0 or len(hits_neg) == 0:
        return None
    d_pos = np.linalg.norm(hits_pos - center, axis=1)
    d_neg = np.linalg.norm(hits_neg - center, axis=1)
    nearest_pos = hits_pos[np.argmin(d_pos)]
    nearest_neg = hits_neg[np.argmin(d_neg)]
    width = float(np.linalg.norm(nearest_pos - nearest_neg))
    return (
        nearest_neg.astype(np.float32),
        nearest_pos.astype(np.float32),
        width,
    )


def _tips_from_anchor(anchor: AnchorRecord) -> tuple[np.ndarray, np.ndarray]:
    if anchor.tip_l is not None and anchor.tip_r is not None:
        return (
            anchor.tip_l.astype(np.float64).copy(),
            anchor.tip_r.astype(np.float64).copy(),
        )
    R = anchor.rotation
    f0 = _unit(R[:, 0])
    center = anchor.grasp_point.astype(np.float64)
    if anchor.finger_width_actual > 0:
        half = anchor.finger_width_actual / 2.0
    else:
        half = max((anchor.gripper_width - GRIPPER_WIDTH_PAD) / 2.0, 0.001)
    return center - f0 * half, center + f0 * half


def _small_so3_matrix(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = float(rng.uniform(0.0, np.deg2rad(max_deg)))
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _perturb_anchor_pose(
    anchor: AnchorRecord,
    rng: np.random.Generator,
    *,
    max_rot_deg: float,
    max_trans_m: float,
    max_approach_trans_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perturb candidate grasp_point + rotation (same semantics as raycast sampler).
    Raycast is done from grasp_point along finger_dir — not from tip midpoint.
    """
    R0 = np.asarray(anchor.rotation, dtype=np.float64).reshape(3, 3)
    R_delta = _small_so3_matrix(rng, max_rot_deg)
    R1 = (R_delta @ R0).astype(np.float32)
    approach = _unit(R1[:, 2]).astype(np.float32)
    finger = _orthogonalize_finger(R1[:, 0], approach).astype(np.float32)
    closure = np.cross(approach, finger)
    cn = np.linalg.norm(closure)
    if cn < 1e-9:
        closure = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        closure = (closure / cn).astype(np.float32)

    jf = float(rng.uniform(-max_trans_m, max_trans_m))
    jc = float(rng.uniform(-max_trans_m, max_trans_m))
    ja = float(rng.uniform(-max_approach_trans_m, max_approach_trans_m))
    delta = finger.astype(np.float64) * jf + closure.astype(np.float64) * jc + approach.astype(
        np.float64,
    ) * ja
    grasp_point = (anchor.grasp_point.astype(np.float64) + delta).astype(np.float32)
    return grasp_point, approach, finger, R1


def try_strategy_a_candidate(
    anchor: AnchorRecord,
    mesh,
    mesh_rc,
    z_min: float,
    z_max: float,
    rng: np.random.Generator,
    *,
    max_rot_deg: float,
    max_tip_jitter_mm: float,
    require_hp_contact: bool,
    hp_pc,
    hp_labels,
    rgs_module,
) -> tuple[dict | None, str | None]:
    grasp_point, approach, finger_dir, R_pert = _perturb_anchor_pose(
        anchor,
        rng,
        max_rot_deg=max_rot_deg,
        max_trans_m=max_tip_jitter_mm / 1000.0,
        max_approach_trans_m=1.0 / 1000.0,
    )
    rc = mesh_rc if mesh_rc is not None else mesh

    ray = _raycast_contacts(rc, grasp_point, finger_dir)
    if ray is None:
        return None, "raycast"
    contact_l, contact_r, width = ray
    if width > rgs_module.MAX_GRIPPER_OPEN or width < rgs_module.MIN_GRIPPER_WIDTH:
        return None, "width_bounds"

    if not rgs_module.passes_hp_contact_requirement(
        contact_l, contact_r, hp_pc, hp_labels, require=require_hp_contact,
    ):
        return None, "hp_contact"

    grasp_center = ((contact_l + contact_r) / 2.0).astype(np.float32)
    cand = rgs_module._build_candidate_dict(
        mesh,
        z_min,
        z_max,
        grasp_center=grasp_center,
        contact_L=contact_l,
        contact_R=contact_r,
        width=width,
        approach=approach.astype(np.float32),
        finger_dir=finger_dir,
        mesh_rc=rc,
    )
    if cand is None:
        return None, "build_candidate"
    cand["rotation"] = R_pert

    if anchor.mesh_prerotation is not None:
        cand["mesh_prerotation"] = anchor.mesh_prerotation
    return cand, None


def generate_anchored_candidates(
    mesh,
    mesh_rc,
    merged_path: str,
    *,
    target_n: int,
    score_threshold: float,
    require_hp_contact: bool,
    hp_pc,
    hp_labels,
    max_rot_deg: float = 8.0,
    max_tip_jitter_mm: float = 3.0,
    max_retry_per_slot: int | None = None,
    filter_trusted: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    import random_grasp_sampler as rgs

    anchors, load_meta = load_anchors_from_merged(
        merged_path,
        filter_trusted=filter_trusted,
        allow_fallback_all=True,
    )
    meta: dict[str, Any] = {
        "sampling_method": SAMPLING_METHOD_ANCHORED,
        "target_n": target_n,
        "score_threshold": score_threshold,
        "require_hp_contact": require_hp_contact,
        "perturb": {
            "max_rot_deg": max_rot_deg,
            "max_tip_jitter_mm": max_tip_jitter_mm,
        },
        "hard_gates": "same_as_raycast_no_dedup",
        **load_meta,
    }

    K = len(anchors)
    if K == 0:
        meta["error"] = "no_anchors"
        return [], meta

    meta["hp_contact_required_effective"] = require_hp_contact

    rng = np.random.default_rng()
    schedule = build_anchor_schedule(K, target_n, rng)
    n_slots = len(schedule)
    retry_per_slot = (
        int(max_retry_per_slot)
        if max_retry_per_slot is not None
        else compute_max_retry_per_slot(target_n, n_slots)
    )
    meta["anchor_schedule_len"] = n_slots
    meta["max_retry_per_slot"] = retry_per_slot
    meta["retry_attempt_budget"] = retry_per_slot * n_slots
    meta["retry_budget_rule"] = "100x_target_over_schedule_slots"
    usage: dict[str, int] = {a.name: 0 for a in anchors}

    z_min = float(mesh.bounds[0][2])
    z_max = float(mesh.bounds[1][2])

    all_candidates: list[dict] = []
    n_attempts = 0
    n_reject = 0
    reject_reasons: dict[str, int] = {}

    for slot_i, anchor_idx in enumerate(schedule):
        anchor = anchors[anchor_idx]
        usage[anchor.name] = usage.get(anchor.name, 0) + 1
        slot_added = 0
        for _ in range(retry_per_slot):
            n_attempts += 1
            cand, reject = try_strategy_a_candidate(
                anchor,
                mesh,
                mesh_rc,
                z_min,
                z_max,
                rng,
                max_rot_deg=max_rot_deg,
                max_tip_jitter_mm=max_tip_jitter_mm,
                require_hp_contact=require_hp_contact,
                hp_pc=hp_pc,
                hp_labels=hp_labels,
                rgs_module=rgs,
            )
            if cand is None:
                n_reject += 1
                if reject:
                    reject_reasons[reject] = reject_reasons.get(reject, 0) + 1
                continue
            cand["name"] = f"anchored_{anchor.name}_{slot_i}_{slot_added}"
            all_candidates.append(cand)
            slot_added += 1

            high_quality = [c for c in all_candidates if c["score"] >= score_threshold]
            if len(high_quality) >= target_n:
                break
        else:
            if slot_added == 0:
                meta.setdefault("failed_slots", []).append(slot_i)
            continue
        high_quality = [c for c in all_candidates if c["score"] >= score_threshold]
        if len(high_quality) >= target_n:
            break

    # Same as raycast: rank by score, take top target_n (may be below threshold).
    all_candidates.sort(key=lambda c: -c["score"])
    selected = all_candidates[:target_n]
    for i, c in enumerate(selected):
        c["name"] = f"anchored_{i}"

    n_high = sum(1 for c in all_candidates if c["score"] >= score_threshold)
    meta["anchor_usage"] = usage
    meta["n_pool"] = len(all_candidates)
    meta["n_high_quality"] = n_high
    meta["n_accepted"] = len(selected)
    meta["n_shortfall"] = max(0, target_n - len(selected))
    meta["n_attempts"] = n_attempts
    meta["n_reject"] = n_reject
    meta["reject_reasons"] = reject_reasons
    meta["score_selection"] = "top_n_by_score_like_raycast"
    if selected:
        print(
            f"  → anchored 选出 {len(selected)} 个 "
            f"≥{score_threshold:.0f}分: {n_high}/{len(selected)}  "
            f"score {selected[0]['score']:.1f}~{selected[-1]['score']:.1f}",
        )
        meta["score_min"] = float(selected[-1]["score"])
        meta["score_max"] = float(selected[0]["score"])
    elif reject_reasons:
        top = sorted(reject_reasons.items(), key=lambda x: -x[1])[:5]
        print(f"  ⚠️  [anchored] reject reasons (top): {top}")
    return selected, meta


__all__ = [
    "SAMPLING_METHOD_ANCHORED",
    "AnchorRecord",
    "is_trusted_grasp",
    "load_anchors_from_merged",
    "build_anchor_schedule",
    "compute_max_retry_per_slot",
    "generate_anchored_candidates",
]

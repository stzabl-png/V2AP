#!/usr/bin/env python3
"""
Visualize GraspNet Top-3 grasps for each eval object.
Generates one PNG per object showing the mesh + gripper poses.
Uses matplotlib 3D for headless rendering (no display needed).
"""

import os
import sys
import csv
import json
import h5py
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Colors for top-3 grippers: rank0=red, rank1=orange, rank2=purple
GRIPPER_COLORS = ["#d62728", "#ff7f0e", "#9467bd"]
GRIPPER_LABELS = ["#1", "#2", "#3"]


def draw_gripper_3d(ax, center, rotation, width=0.08, depth=0.04, finger_len=0.04,
                    color="red", alpha=1.0):
    """Draw a parallel-jaw gripper in 3D as lines + surfaces."""
    finger_dir = rotation[:, 0]
    approach = rotation[:, 2]

    base = center - approach * depth
    left = base + finger_dir * (width / 2)
    right = base - finger_dir * (width / 2)
    left_tip = left + approach * finger_len
    right_tip = right + approach * finger_len

    lw = 2.5 * alpha
    ax.plot(*zip(left, right), color=color, linewidth=lw, alpha=alpha)
    ax.plot(*zip(left, left_tip), color=color, linewidth=lw, alpha=alpha)
    ax.plot(*zip(right, right_tip), color=color, linewidth=lw, alpha=alpha)
    ax.plot(*zip(base, center), color=color, linewidth=1.5, linestyle="--", alpha=alpha)
    ax.scatter(*center, color=color, s=30, zorder=5, alpha=alpha)


def render_object_grasp(
    mesh_path: str,
    scale_factor: float,
    grasps: list,  # list of dicts with position, rotation, score
    obj_id: str,
    output_path: str,
):
    """Render one object with its top-3 grasps and save PNG."""
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = mesh.vertices * scale_factor  # metric
    faces = mesh.faces

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample vertices for plotting (too many is slow)
    n_pts = min(5000, len(verts))
    idx = np.random.choice(len(verts), n_pts, replace=False)
    pts = verts[idx]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="steelblue", s=0.3, alpha=0.4)

    if grasps:
        # Draw grippers: later ranks drawn first (behind), rank 0 on top
        for gi in range(len(grasps) - 1, -1, -1):
            g = grasps[gi]
            alpha = 1.0 if gi == 0 else 0.6
            draw_gripper_3d(
                ax, g["position"], g["rotation"],
                width=0.08, depth=0.04,
                color=GRIPPER_COLORS[gi], alpha=alpha,
            )
        score_strs = [f'{GRIPPER_LABELS[i]}={grasps[i]["score"]:.2f}' for i in range(len(grasps))]
        status = "  ".join(score_strs)
    else:
        status = "NO CANDIDATE (collision filtered)"

    # ── Draw coordinate axes at mesh center ──
    # Matches IsaacSim world frame: Red=X(left/right), Green=Y(front/back), Blue=Z(up)
    origin = (verts.max(0) + verts.min(0)) / 2
    extents = verts.max(0) - verts.min(0)
    ax_len = max(extents) * 0.4  # axis arrow length

    for axis_idx, (color, label, sign) in enumerate([
        ("red",   "+X", np.array([1, 0, 0])),
        ("green", "+Y", np.array([0, 1, 0])),
        ("blue",  "+Z (up)", np.array([0, 0, 1])),
    ]):
        tip = origin + sign * ax_len
        ax.quiver(
            origin[0], origin[1], origin[2],
            sign[0] * ax_len, sign[1] * ax_len, sign[2] * ax_len,
            color=color, linewidth=2.5, arrow_length_ratio=0.15,
        )
        ax.text(
            tip[0], tip[1], tip[2], f" {label}",
            color=color, fontsize=9, fontweight="bold",
        )

    # Draw table plane at Z = mesh bottom (matches collision detection)
    z_bottom = verts[:, 2].min()
    table_margin = max(extents[:2]) * 0.3
    x_lo, x_hi = verts[:, 0].min() - table_margin, verts[:, 0].max() + table_margin
    y_lo, y_hi = verts[:, 1].min() - table_margin, verts[:, 1].max() + table_margin
    n_table = 2000
    tx = np.random.uniform(x_lo, x_hi, n_table)
    ty = np.random.uniform(y_lo, y_hi, n_table)
    tz = np.full(n_table, z_bottom)
    ax.scatter(tx, ty, tz, c="lightgray", s=0.3, alpha=0.5)
    ax.text(x_hi, y_hi, z_bottom, " table", color="gray", fontsize=8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)  [up ↑]")
    ax.set_title(f"{obj_id}\n{status}", fontsize=14, fontweight="bold")

    # Equal aspect ratio
    center = origin
    max_extent = max(extents) * 0.7
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - max_extent, c + max_extent)

    # View from "robot perspective": looking from -Y toward +Y, Z is up
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    eval_csv = os.path.join(PROJ, "evaluation", "configs", "eval_objects_merged_success_ge30.csv")
    cand_dir = os.path.join(PROJ, "output", "graspnet_candidates")
    out_dir = os.path.join(PROJ, "output", "graspnet_vis")
    os.makedirs(out_dir, exist_ok=True)

    with open(eval_csv) as f:
        obj_ids = [r["obj_id"] for r in csv.DictReader(f)]

    print(f"Generating {len(obj_ids)} visualizations...")
    results = []

    for i, oid in enumerate(obj_ids):
        if oid.startswith("ycb_dex_"):
            ds = "ycb"
        elif oid.startswith("unseen_"):
            ds = "unseen"
        else:
            ds = "oakink"
        mesh_path = os.path.join(
            PROJ, "data_hub", "meshes", "SAM3DMesh", "rotated_mesh", ds, oid, "mesh.ply"
        )
        scale_path = os.path.join(
            PROJ, "data_hub", "ProcessedData", "obj_meshes", ds, oid, "scale.json"
        )
        hdf5_path = os.path.join(cand_dir, oid, "trial_00_grasp.hdf5")

        if not os.path.exists(mesh_path):
            print(f"  [{i+1}/{len(obj_ids)}] {oid}: mesh not found")
            continue

        with open(scale_path) as f:
            sf = json.load(f)["scale_factor"]

        # Read top-3 from HDF5
        grasps = []
        if os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, "r") as hf:
                n = hf["candidates"].attrs.get("n_candidates", 0)
                for ci in range(min(n, 3)):
                    c = hf["candidates"][f"candidate_{ci}"]
                    grasps.append({
                        "position": c["position"][:],
                        "rotation": c["rotation"][:],
                        "score": float(c.attrs.get("score", 0)),
                    })

        out_path = os.path.join(out_dir, f"{oid}.png")
        render_object_grasp(mesh_path, sf, grasps, oid, out_path)

        if grasps:
            score_str = ", ".join(f"{g['score']:.2f}" for g in grasps)
            status = f"{len(grasps)} grasps [{score_str}]"
        else:
            status = "EMPTY"
        print(f"  [{i+1}/{len(obj_ids)}] {oid}: {status} → {os.path.basename(out_path)}")
        results.append({"obj_id": oid, "empty": len(grasps) == 0, "score": grasps[0]['score'] if grasps else 0})

    # Summary
    ok = sum(1 for r in results if not r["empty"])
    empty = sum(1 for r in results if r["empty"])
    print(f"\n✅ Done: {ok} with grasps, {empty} empty → {out_dir}/")


if __name__ == "__main__":
    main()

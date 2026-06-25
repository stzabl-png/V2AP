#!/usr/bin/env python3
"""
vis_grasp_result.py — 统一抓取结果可视化工具
=============================================

整合了以下脚本:
  • analysis/vis_grasp_candidates.py
  • analysis/vis_grasp_pose.py
  • tools/vis_robot_gt.py
  • analysis/vis_combined.py
  • tools/vis_3panel.py

用法:
    # 单物体 — 3列汇报图 (HP | Robot GT | Prediction)
    python tools/vis_grasp_result.py --obj A16013

    # 抓取位姿图 (mesh + 热力图 + 夹爪 + Force Center)
    python tools/vis_grasp_result.py --obj A16013 --mode grasp

    # Robot GT 可视化 (Open3D 交互)
    python tools/vis_grasp_result.py --obj A16013 --mode robot_gt

    # 3列综合图 (物体渲染 | 候选抓取 | Affordance+夹爪)
    python tools/vis_grasp_result.py --obj A16013 --mode combined

    # 批量输出 3panel PNG
    python tools/vis_grasp_result.py --batch --out output/vis_grasp

    # 汇总统计 Robot GT 状态
    python tools/vis_grasp_result.py --summary
"""

import os, sys, glob, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "model"))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINING_FP_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "training_fp")
MESH_V1_DIR     = config.MESH_V1_DIR
GRASP_DIR       = config.GRASPS_DIR
OUT_BASE        = os.path.join(PROJ, "output", "vis_grasp_result")
GT_DIRS = [
    os.path.join(PROJ, "output", "robot_gt_verified"),
    os.path.join(PROJ, "output", "robot_gt_v1_manual"),
    os.path.join(PROJ, "output", "robot_gt_v2_raycast"),
    os.path.join(PROJ, "output", "robot_gt"),
]
CKPT_SEARCH = [
    os.path.join(PROJ, "output", "checkpoints_m5", "best_m5_model.pth"),
    os.path.join(PROJ, "output", "checkpoints",    "best_model.pth"),
]
N_DENSE  = 20000
FRANKA_MESH_DIR = os.path.join(PROJ, "sim", "assets_franka")

CANDIDATE_COLORS = {
    "horizontal_front": "#e74c3c",
    "horizontal_right": "#2980b9",
    "horizontal_left":  "#f39c12",
    "top_down":         "#8e44ad",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_mesh(obj_id):
    for ext in (".obj", ".ply"):
        p = os.path.join(MESH_V1_DIR, f"{obj_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_training_data(obj_id, dataset="dexycb"):
    for base_ds in [dataset, ""]:
        path = os.path.join(TRAINING_FP_DIR, base_ds, f"{obj_id}.hdf5")
        if not os.path.exists(path):
            path = os.path.join(TRAINING_FP_DIR, f"{obj_id}.hdf5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                return {
                    "pc":  f["point_cloud"][()].astype(np.float32),
                    "nrm": f["normals"][()].astype(np.float32),
                    "hp":  f["human_prior"][()].astype(np.float32),
                    "rgt": f["robot_gt"][()].astype(np.float32) if "robot_gt" in f else None,
                    "fc":  f["force_center"][()].astype(np.float32) if "force_center" in f else None,
                }
    return None


def load_grasp_hdf5(obj_id):
    path = os.path.join(GRASP_DIR, f"{obj_id}_grasp.hdf5")
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        d = {
            "points":       f["affordance/points"][:],
            "probs":        f["affordance/contact_prob"][:],
            "grasp_pos":    f["grasp/position"][:],
            "grasp_rot":    f["grasp/rotation"][:],
            "gripper_width":f["grasp"].attrs["gripper_width"],
            "best_name":    f["grasp"].attrs.get("candidate_name", ""),
            "force_center": f["affordance/force_center"][:] if "affordance/force_center" in f else None,
            "grasp_point":  f["grasp/grasp_point"][:] if "grasp/grasp_point" in f else None,
            "candidates":   [],
        }
        if "candidates" in f:
            n = f["candidates"].attrs.get("n_candidates", 0)
            for i in range(n):
                cg = f[f"candidates/candidate_{i}"]
                d["candidates"].append({
                    "name":      str(cg.attrs["name"]),
                    "position":  cg["position"][:],
                    "grasp_point": cg["grasp_point"][:] if "grasp_point" in cg else cg["position"][:],
                    "rotation":  cg["rotation"][:],
                    "gripper_width":       float(cg.attrs["gripper_width"]),
                    "cross_section_width": float(cg.attrs["cross_section_width"]),
                    "score":     float(cg.attrs["score"]),
                })
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ══════════════════════════════════════════════════════════════════════════════

def _set_ax(ax, pts, scale=0.6):
    r = (pts.max(0) - pts.min(0)).max() * scale
    c = (pts.max(0) + pts.min(0)) / 2
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r); ax.set_axis_off()


def render_pc(ax, pts, vals, title, cmap="jet", binary=False,
              elev=25, azim=135, fc=None):
    order = np.argsort(vals); p, v = pts[order], vals[order]
    if binary:
        colors = np.where(v[:, None] >= 0.5,
                          [[0.9, 0.15, 0.15, 1.0]], [[0.75, 0.75, 0.85, 1.0]])
    else:
        colors = plt.get_cmap(cmap)(np.clip(v, 0, 1))
    ax.scatter(p[:,0], p[:,1], p[:,2], c=colors, s=1.5, alpha=0.9, edgecolors="none")
    if fc is not None:
        ax.scatter(*fc, c="lime", s=120, marker="*",
                   edgecolors="darkgreen", linewidth=1.2, zorder=10, label="force center")
        ax.legend(fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.view_init(elev=elev, azim=azim); _set_ax(ax, p)


def dense_interpolate(mesh_path, pc, *arrays):
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    vis_pc, _ = trimesh.sample.sample_surface(mesh, N_DENSE)
    vis_pc = vis_pc.astype(np.float32)
    _, idx = cKDTree(pc).query(vis_pc, k=3)
    dists  = np.linalg.norm(vis_pc[:,None,:] - pc[idx], axis=2)
    w = 1.0/(dists+1e-8); w /= w.sum(1, keepdims=True)
    res = [vis_pc] + [(w * a[idx]).sum(1) for a in arrays]
    return res


def render_mesh_panel(ax, mesh_path, elev=15, azim=-60):
    """Panel 1: solid shaded mesh."""
    import trimesh
    mesh  = trimesh.load(mesh_path, force="mesh")
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    step  = max(1, len(faces)//3000)
    faces = faces[::step]
    v0,v1,v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    normals = np.cross(v1-v0, v2-v0)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    ld = np.array([0.3,-0.5,0.8]); ld /= np.linalg.norm(ld)
    intensity = np.clip(normals @ ld, 0.15, 1.0)
    fc_col = np.outer(intensity, np.array([0.82,0.82,0.82]))
    fc_col = np.hstack([np.clip(fc_col,0,1), np.ones((len(fc_col),1))*0.95])
    poly = Poly3DCollection(verts[faces])
    poly.set_facecolor(fc_col); poly.set_edgecolor((0.35,0.35,0.35,0.2))
    poly.set_linewidth(0.15); ax.add_collection3d(poly)
    _set_ax(ax, verts); ax.view_init(elev=elev, azim=azim)


def draw_gripper(ax, grasp_pt, rot, width, color="#333333"):
    xo, yb, za = rot[:,0], rot[:,1], rot[:,2]
    hw = width/2; fl = 0.04; fw = 0.004
    for sign in [1,-1]:
        tip  = grasp_pt + sign*xo*(hw-fw)
        base = tip - za*fl
        c1,c2,c3,c4 = base-yb*fw, base+yb*fw, tip+yb*fw, tip-yb*fw
        for a,b in [(c1,c2),(c2,c3),(c3,c4),(c4,c1)]:
            ax.plot3D([a[0],b[0]],[a[1],b[1]],[a[2],b[2]], c=color, lw=1.5, alpha=0.8)
    cb = grasp_pt - za*fl
    lb,rb = cb+xo*hw, cb-xo*hw
    ax.plot3D([lb[0],rb[0]],[lb[1],rb[1]],[lb[2],rb[2]], c=color, lw=2, alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════════
# Mode: 3panel  (HP | Robot GT | Model Prediction)
# ══════════════════════════════════════════════════════════════════════════════

def mode_3panel(obj_id, dataset, model, device, out_path):
    data = load_training_data(obj_id, dataset)
    if data is None:
        print(f"  ⚠️ 无训练数据: {obj_id}"); return False

    pc, hp, rgt, fc = data["pc"], data["hp"], data["rgt"], data["fc"]

    # model inference
    pred = None
    if model is not None:
        import torch
        feats = np.concatenate([pc, data["nrm"], hp[:,None]], axis=-1)
        with torch.no_grad():
            pred = model(
                torch.from_numpy(pc).unsqueeze(0).to(device),
                torch.from_numpy(feats).unsqueeze(0).to(device),
            ).squeeze(0).cpu().numpy().ravel()

    mesh_path = find_mesh(obj_id)
    if mesh_path:
        arrays = [hp] + ([rgt] if rgt is not None else []) + ([pred] if pred is not None else [])
        res = dense_interpolate(mesh_path, pc, *arrays)
        vis_pc = res[0]; idx = 1
        hp_d  = res[idx]; idx+=1
        rgt_d = res[idx] if rgt is not None else None; idx += int(rgt is not None)
        pred_d= res[idx] if pred is not None else None
    else:
        vis_pc, hp_d, rgt_d, pred_d = pc, hp, rgt, pred

    n_panels = 1 + (1 if rgt_d is not None else 0) + (1 if pred_d is not None else 0)
    n_panels = max(n_panels, 2)

    fig = plt.figure(figsize=(7*n_panels, 7), facecolor="white")
    fig.suptitle(f"{obj_id}  [{dataset}]", fontsize=14, fontweight="bold", y=0.98)

    col = 1
    ax1 = fig.add_subplot(1, n_panels, col, projection="3d"); col+=1
    render_pc(ax1, vis_pc, hp_d, "Human Prior", binary=True, fc=fc)

    if rgt_d is not None:
        ax2 = fig.add_subplot(1, n_panels, col, projection="3d"); col+=1
        render_pc(ax2, vis_pc, rgt_d, "Robot GT", fc=fc)

    if pred_d is not None:
        ax3 = fig.add_subplot(1, n_panels, col, projection="3d")
        render_pc(ax3, vis_pc, pred_d, f"Model Pred\nmax={pred_d.max():.3f}", fc=fc)
    elif rgt_d is None:
        ax2 = fig.add_subplot(1, n_panels, col, projection="3d")
        render_pc(ax2, vis_pc, hp_d, "Human Prior (jet)", cmap="jet", fc=fc)

    plt.tight_layout(rect=[0,0,1,0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → {out_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Mode: combined  (mesh | candidates | affordance+gripper)
# ══════════════════════════════════════════════════════════════════════════════

def mode_combined(obj_id, out_path):
    mesh_path = find_mesh(obj_id)
    if not mesh_path:
        print(f"  ❌ mesh 不存在: {obj_id}"); return False
    grasp = load_grasp_hdf5(obj_id)
    if grasp is None:
        print(f"  ❌ grasp HDF5 不存在: {obj_id}"); return False

    pts, probs = grasp["points"], grasp["probs"]
    fc   = grasp["force_center"]
    rot  = grasp["grasp_rot"]
    gpos = grasp["grasp_pos"]
    gw   = grasp["gripper_width"]

    # dense affordance
    res     = dense_interpolate(mesh_path, pts, probs)
    dpts, dp = res[0], res[1]

    fig = plt.figure(figsize=(21,7), facecolor="white")
    fig.suptitle(f"{obj_id}  —  {len(grasp['candidates'])} candidates",
                 fontsize=14, fontweight="bold", y=0.97)

    # Panel 1: mesh
    ax1 = fig.add_subplot(131, projection="3d", facecolor="white")
    render_mesh_panel(ax1, mesh_path)
    ax1.set_title("3D Object Model", fontsize=12); ax1.set_axis_off()

    # Panel 2: candidates
    ax2 = fig.add_subplot(132, projection="3d", facecolor="white")
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = np.array(mesh.vertices)
    # transparent mesh
    faces = np.array(mesh.faces)[::max(1,len(mesh.faces)//3000)]
    poly = Poly3DCollection(verts[faces], alpha=0.10, facecolor="lightskyblue",
                             edgecolor="gray", linewidth=0.1)
    ax2.add_collection3d(poly)
    if fc is not None:
        ax2.scatter(*fc, c="lime", s=200, marker="*",
                    edgecolors="darkgreen", linewidth=1.2, zorder=10)
    best_name = grasp["best_name"]
    for i, c in enumerate(grasp["candidates"]):
        color = CANDIDATE_COLORS.get(c["name"], "#888888")
        gp    = c["grasp_point"]
        draw_gripper(ax2, gp, c["rotation"], c["gripper_width"], color=color)
        ax2.scatter(*gp, c=color, s=80, marker="+", linewidth=2.5, zorder=10,
                    label=f"{c['name']} {'⭐' if c['name']==best_name else ''}")
        if fc is not None:
            ax2.plot3D(*zip(fc, gp), "--", color=color, lw=1, alpha=0.5)
    _set_ax(ax2, verts); ax2.view_init(15, -60)
    ax2.legend(fontsize=6); ax2.set_title(f"Grasp Candidates ({len(grasp['candidates'])})", fontsize=12)

    # Panel 3: heatmap + best gripper
    ax3 = fig.add_subplot(133, projection="3d", facecolor="white")
    order = np.argsort(dp)
    ax3.scatter(dpts[order,0], dpts[order,1], dpts[order,2],
                c=cm.jet(dp[order]), s=3, alpha=0.9, edgecolors="none")
    draw_gripper(ax3, gpos, rot, gw)
    if fc is not None:
        ax3.scatter(*fc, c="lime", s=150, marker="*",
                    edgecolors="darkgreen", linewidth=1.2, zorder=10)
    _set_ax(ax3, dpts); ax3.view_init(15, -60)
    ax3.set_title("Affordance + Gripper", fontsize=12)

    sm = cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(0,1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.65])
    fig.colorbar(sm, cax=cbar_ax).set_label("Contact Probability", fontsize=10)
    plt.subplots_adjust(left=0.01, right=0.91, wspace=0.02, top=0.90, bottom=0.03)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → {out_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Mode: robot_gt  (Open3D interactive)
# ══════════════════════════════════════════════════════════════════════════════

def mode_robot_gt(obj_id):
    import open3d as o3d, trimesh
    mesh_path = find_mesh(obj_id)
    if not mesh_path:
        print(f"❌ mesh 不存在: {obj_id}"); return

    mesh = trimesh.load(mesh_path, force="mesh")
    grasp_data = []
    for gt_dir in GT_DIRS:
        gt_path = os.path.join(gt_dir, f"{obj_id}_robot_gt.hdf5")
        if not os.path.exists(gt_path): continue
        with h5py.File(gt_path, "r") as f:
            if not f.attrs.get("success", False): continue
            if "successful_grasps" not in f: continue
            for key in f["successful_grasps"].keys():
                g   = f[f"successful_grasps/{key}"]
                gp  = g["grasp_point"][:]
                ad  = g["approach_dir"][:] if "approach_dir" in g else None
                fd  = g["finger_dir"][:] if "finger_dir" in g else None
                w   = g.attrs.get("gripper_width", 0.04)
                if ad is None or fd is None: continue
                mid = gp + ad * 0.105
                c1 = c2 = None
                for d, di in [(fd, 1), (-fd, -1)]:
                    locs, _, _ = mesh.ray.intersects_location([mid], [d])
                    if len(locs):
                        c = locs[np.argmin(np.linalg.norm(locs-mid, axis=1))]
                        if di == 1: c1 = c
                        else:       c2 = c
                grasp_data.append({"mid": mid, "c1": c1, "c2": c2})
        if grasp_data: break

    if not grasp_data:
        print(f"⚠️ {obj_id} 无成功 Robot GT"); return

    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d_mesh.compute_vertex_normals()
    wf = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
    wf.paint_uniform_color([0.5,0.5,0.5])
    geoms = [wf]
    for gd in grasp_data:
        sp = o3d.geometry.TriangleMesh.create_sphere(0.003)
        sp.translate(gd["mid"]); sp.paint_uniform_color([0,0.3,1]); geoms.append(sp)
        for cp in [gd["c1"], gd["c2"]]:
            if cp is None: continue
            s = o3d.geometry.TriangleMesh.create_sphere(0.003)
            s.translate(cp); s.paint_uniform_color([1,0,0]); geoms.append(s)
    print(f"🖱️  左键=旋转  滚轮=缩放  Q=退出  ({len(grasp_data)} grasps)")
    o3d.visualization.draw_geometries(geoms,
        window_name=f"Robot GT — {obj_id}", width=1200, height=800)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def mode_summary():
    all_meshes = sorted(glob.glob(os.path.join(MESH_V1_DIR, "*.obj")))
    results = {"success": [], "fail": [], "missing": []}
    for mp in all_meshes:
        obj_id = os.path.splitext(os.path.basename(mp))[0]
        status = "missing"
        for gt_dir in GT_DIRS:
            p = os.path.join(gt_dir, f"{obj_id}_robot_gt.hdf5")
            if os.path.exists(p):
                try:
                    with h5py.File(p, "r") as f:
                        status = "success" if f.attrs.get("success") else "fail"
                        if status == "success": break
                except: pass
        results[status].append(obj_id)
    total = len(all_meshes)
    print("="*55)
    print(f"  ✅ 成功: {len(results['success'])}/{total}")
    print(f"  ❌ 失败: {len(results['fail'])}/{total}")
    print(f"  🆕 缺失: {len(results['missing'])}/{total}")
    if results["success"]:
        print(f"\n✅ 成功: {', '.join(results['success'])}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="统一抓取结果可视化 (vis_grasp_result.py)")
    p.add_argument("--obj",     default=None)
    p.add_argument("--dataset", default="dexycb",
                   choices=["dexycb", "ho3d_v3"])
    p.add_argument("--mode",    default="3panel",
                   choices=["3panel", "combined", "robot_gt"],
                   help="3panel=HP|GT|Pred  combined=mesh|候选|热力图  robot_gt=Open3D")
    p.add_argument("--ckpt",    default=None)
    p.add_argument("--batch",   action="store_true")
    p.add_argument("--summary", action="store_true", help="Robot GT 汇总统计")
    p.add_argument("--out",     default=OUT_BASE)
    args = p.parse_args()

    if args.summary:
        mode_summary(); return

    model = device = None
    if args.mode == "3panel":
        ckpt = args.ckpt
        for c in CKPT_SEARCH:
            if not ckpt and os.path.exists(c): ckpt = c; break
        if ckpt:
            import torch
            from pointnet2 import PointNet2Seg
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = PointNet2Seg(num_classes=1, in_channel=7).to(device)
            ck     = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(ck.get("model_state_dict", ck))
            model.eval()
            print(f"✅ Model loaded from {os.path.basename(ckpt)}")
        else:
            print("⚠️  无 checkpoint，3panel 将跳过预测列")

    def run_one(obj_id):
        if args.mode == "3panel":
            out = os.path.join(args.out, "3panel", f"{obj_id}.png")
            return mode_3panel(obj_id, args.dataset, model, device, out)
        elif args.mode == "combined":
            out = os.path.join(args.out, "combined", f"{obj_id}.png")
            return mode_combined(obj_id, out)
        elif args.mode == "robot_gt":
            mode_robot_gt(obj_id); return True

    if args.batch:
        folder = os.path.join(MESH_V1_DIR)
        objs   = sorted(os.path.splitext(f)[0]
                        for f in os.listdir(folder) if f.endswith(".obj"))
        print(f"批量 {len(objs)} 个物体 [mode={args.mode}]")
        ok = sum(1 for obj in objs if (print(f"\n  [{obj}]") or True) and run_one(obj))
        print(f"\n✅ {ok}/{len(objs)} 完成 → {args.out}/")
    elif args.obj:
        run_one(args.obj)
    else:
        print("请指定 --obj <名称> 或 --batch 或 --summary")


if __name__ == "__main__":
    main()

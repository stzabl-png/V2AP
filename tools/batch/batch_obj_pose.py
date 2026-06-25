#!/usr/bin/env python3
"""
batch_obj_pose.py — 通用物体位姿估计脚本 (FoundationPose)

处理流程（每条序列）：
  RGB 帧 + Depth Pro 深度 + SAM3D Mesh + SAM2 Mask
    → 准备 scene dir（color/ depth/ masks/ cam_K.txt）
    → FoundationPose register (第0帧) + track (后续帧)
    → 保存 ob_in_cam/{frame_id}.txt (4×4 位姿矩阵)
    → 保存 track_vis/{frame_id}.png (可视化)

输出结构:
  data_hub/ProcessedData/obj_poses/{dataset}/{seq_id}/
    ob_in_cam/000001.txt  ...
    track_vis/000001.png  ...

用法:
  conda activate bundlesdf
  cd /home/lyh/Project/V2AP

  # 验证：跑 3 个 arctic/box 序列
  python tools/batch_obj_pose.py --dataset arctic --limit 3

  # 全量
  python tools/batch_obj_pose.py
  python tools/batch_obj_pose.py --dataset oakink
  python tools/batch_obj_pose.py --seq box_use_01    # seq_id 过滤

物体-Mesh 映射规则:
  arctic:  s01__box_use_01   → obj_meshes/arctic/box/mesh.ply
  oakink:  A01001_0001_0000  → obj_meshes/oakink/A01001/mesh.ply
  ho3d_v3: ABF10             → obj_meshes/ycb/003_cracker_box/mesh.ply
  dexycb:  s01__20200709__836 → obj_meshes/ycb/ycb_dex_XX/mesh.ply  (TODO: 精确映射)
"""

import os, sys, re, argparse, json, time, shutil
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── FoundationPose 路径 ───────────────────────────────────────────────────────
FP_ROOT = config.FP_ROOT
sys.path.insert(0, FP_ROOT)

# ── 数据路径 ──────────────────────────────────────────────────────────────────
MESH_BASE   = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes")
DEPTH_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "depth", "ThirdPerson")
RAW_BASE    = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
OBJ_INPUT   = os.path.join(config.DATA_HUB, "ProcessedData", "mask", "ThirdPerson")
OUT_BASE    = os.path.join(config.DATA_HUB, "ProcessedData", "poses", "ThirdPerson")
SCENE_TMP   = "/tmp/fp_scenes"

SHORTER_SIDE = 480   # downscale to fit in GPU memory (OOM if 1000×1000)

# ── HO3D prefix → YCB 物体名 ─────────────────────────────────────────────────
HO3D_OBJ = {
    'ABF': '003_cracker_box', 'BB': '011_banana',
    'GPMF': '010_potted_meat_can', 'GSF': '010_potted_meat_can',
    'MC': '003_cracker_box', 'MDF': '035_power_drill',
    'ND': '035_power_drill', 'SB': '021_bleach_cleanser',
    'ShSu': '004_sugar_box', 'SiBF': '003_cracker_box',
    'SiS': '052_extra_large_clamp', 'SM': '006_mustard_bottle',
    'SMu': '006_mustard_bottle', 'SS': '004_sugar_box',
}

# ── DexYCB: fixed per-subject object ordering (from dex-ycb-toolkit) ──────────
DEXYCB_YCB_CLASSES = [
    '002_master_chef_can', '003_cracker_box', '004_sugar_box',
    '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can',
    '008_pudding_box', '009_gelatin_box', '010_potted_meat_can',
    '011_banana', '019_pitcher_base', '021_bleach_cleanser',
    '024_bowl', '025_mug', '035_power_drill',
    '036_wood_block', '037_scissors', '040_large_marker',
    '051_large_clamp', '052_extra_large_clamp',
]
DEXYCB_NUM_OBJECTS = len(DEXYCB_YCB_CLASSES)   # 20
_dexycb_session_cache: dict = {}

def _dexycb_sessions(subj: str):
    if subj not in _dexycb_session_cache:
        subj_dir = os.path.join(RAW_BASE, 'dexycb', subj)
        if os.path.isdir(subj_dir):
            _dexycb_session_cache[subj] = natsorted(
                [d for d in os.listdir(subj_dir)
                 if os.path.isdir(os.path.join(subj_dir, d))
                 and not d.startswith('.')]   # BUG-04: 过滤 .cache 等隐藏目录
            )
        else:
            _dexycb_session_cache[subj] = []
    return _dexycb_session_cache[subj]

def dexycb_seq_to_mesh(seq_id: str):
    """DexYCB seq_id (subj__session__serial) → 'ycb_dex_NN' mesh name."""
    parts = seq_id.split('__')
    if len(parts) != 3:
        return None
    subj, session, _ = parts
    sessions = _dexycb_sessions(subj)
    if not sessions or session not in sessions:
        return None
    idx = sessions.index(session)
    grasps_per_obj = max(1, len(sessions) // DEXYCB_NUM_OBJECTS)
    obj_idx = min(idx // grasps_per_obj, DEXYCB_NUM_OBJECTS - 1)
    return f'ycb_dex_{obj_idx + 1:02d}'


# ── 序列 → (mesh_ds, obj_name) 映射 ─────────────────────────────────────────
def seq_to_obj(dataset, seq_id):
    """Return (mesh_dataset, obj_name) for a given sequence."""
    if dataset == "arctic":
        m = re.match(r'^[^_]+__(.+?)_(grab|use)_', seq_id)
        if m:
            return "arctic", m.group(1)
    elif dataset == "oakink":
        return "oakink", seq_id.split("_")[0]
    elif dataset == "ho3d_v3":
        # BUG-06 修复: seq_id 带有 'train__' / 'evaluation__' 前缀，
        # 必须先剥离才能匹配 HO3D_OBJ 字典
        sid = seq_id.split("__", 1)[-1]   # 'train__ABF10' → 'ABF10'
        prefix = re.sub(r'\d+$', '', sid)  # 'ABF10' → 'ABF'
        ycb = HO3D_OBJ.get(prefix)
        if ycb: return "ycb", ycb
    elif dataset == "dexycb":
        obj_name = dexycb_seq_to_mesh(seq_id)
        if obj_name:
            return "ycb", obj_name
    return None, None


# ── 序列 RGB 帧发现 ───────────────────────────────────────────────────────────
def find_rgb_frames(dataset, seq_id):
    if dataset == "arctic":
        subj, seq = seq_id.split("__", 1)
        d = os.path.join(RAW_BASE, "arctic", subj, seq, "1")
        return natsorted(glob(os.path.join(d, "*.jpg")))
    elif dataset == "oakink":
        d = os.path.join(RAW_BASE, "oakink_v1", seq_id)
        return natsorted(glob(os.path.join(d, "*", "north_west_color_*.png")) +
                         glob(os.path.join(d, "north_west_color_*.png")))
    elif dataset == "ho3d_v3":
        split, sid = seq_id.split("__", 1)
        d = os.path.join(RAW_BASE, "ho3d_v3", split, sid, "rgb")
        return natsorted(glob(os.path.join(d, "*.jpg")))
    elif dataset == "dexycb":
        parts = seq_id.split("__")
        if len(parts) == 3:
            d = os.path.join(RAW_BASE, "dexycb", *parts)
            return natsorted(glob(os.path.join(d, "color_*.jpg")))
    return []


# ── 准备 scene dir ────────────────────────────────────────────────────────────
def prepare_scene(dataset, seq_id, obj_ds, obj_name, scene_dir, shorter_side=480):
    """
    Build scene dir from depth npz + masked annotation.
    Returns downscale factor or None on error.
    """
    depth_dir = os.path.join(DEPTH_BASE, dataset, seq_id)
    npz_path  = os.path.join(depth_dir, "depths.npz")
    fid_path  = os.path.join(depth_dir, "frame_ids.txt")
    K_path    = os.path.join(depth_dir, "K.txt")

    if not os.path.exists(npz_path):
        return None
    data      = np.load(npz_path)
    depths    = data["depths"].astype(np.float32)
    frame_ids = open(fid_path).read().strip().split("\n")
    K_orig    = np.loadtxt(K_path)

    ann_mask = os.path.join(OBJ_INPUT, obj_ds, obj_name, "0.png")
    if not os.path.exists(ann_mask):
        return None

    all_rgb    = find_rgb_frames(dataset, seq_id)
    rgb_by_name = {os.path.basename(p): p for p in all_rgb}

    # Only frames with matching depth
    valid_pairs = [(rgb_by_name[os.path.basename(fid)], i)
                   for i, fid in enumerate(frame_ids)
                   if os.path.basename(fid) in rgb_by_name]
    if not valid_pairs:
        return None

    os.makedirs(os.path.join(scene_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "masks"), exist_ok=True)

    # Determine downscale from first image
    first_rgb = np.array(Image.open(valid_pairs[0][0]))
    H0, W0 = first_rgb.shape[:2]
    scale = shorter_side / min(H0, W0)
    H = int(H0 * scale); W = int(W0 * scale)
    K = K_orig.copy(); K[:2] *= scale

    for i, (rgb_path, dep_idx) in enumerate(valid_pairs):
        out_id = f"{i:06d}"
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        rgb_small = cv2.resize(rgb, (W, H))
        cv2.imwrite(os.path.join(scene_dir, "rgb", f"{out_id}.png"),
                    cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))

        depth_m = cv2.resize(depths[dep_idx], (W, H))
        depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(scene_dir, "depth", f"{out_id}.png"), depth_mm)

    # Mask for frame 0
    mask_raw = cv2.imread(ann_mask, 0)
    mask_s   = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(scene_dir, "masks", "000000.png"), mask_s)

    # Write downscaled K
    np.savetxt(os.path.join(scene_dir, "cam_K.txt"), K, fmt="%.6f")

    return len(valid_pairs), H, W


# ── FoundationPose 推理 ───────────────────────────────────────────────────────
def init_fp_models():
    """Initialize scorer, refiner, glctx ONCE and reuse across sequences."""
    import nvdiffrast.torch as dr
    from estimater import ScorePredictor, PoseRefinePredictor, set_logging_format, set_seed
    set_seed(0)
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    return scorer, refiner, glctx


def run_fp(mesh_path, scene_dir, out_dir, scorer, refiner, glctx,
           est_iter=5, track_iter=2, debug=1):
    import trimesh, torch
    import imageio
    from estimater import (FoundationPose, set_logging_format,
                           draw_posed_3d_box, draw_xyz_axis)
    from datareader import YcbineoatReader

    mesh = trimesh.load(mesh_path)

    # SAM3D meshes can have 500k+ faces → OOM in nvdiffrast batch rendering.
    # Simplify to ≤5000 faces by calling fast_simplification directly.
    if len(mesh.faces) > 5000:
        import fast_simplification
        n_before = len(mesh.faces)
        ratio = 1.0 - min(5000 / n_before, 0.9999)
        pts_out, faces_out = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_reduction=ratio)
        mesh = trimesh.Trimesh(vertices=pts_out, faces=faces_out, process=False)
        print(f"    Mesh simplified: {n_before:,} → {len(mesh.faces):,} faces")

    # Apply scale factor from estimate_obj_scale.py (scale.json next to mesh.ply)
    scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
    if os.path.exists(scale_json):
        with open(scale_json) as f:
            scale_info = json.load(f)
        sf = float(scale_info.get("scale_factor", 1.0))
        if abs(sf - 1.0) > 0.01:
            mesh.vertices *= sf
            print(f"    Scale applied: ×{sf:.4f}  diameter={mesh.bounding_sphere.primitive.radius*2:.3f}m")
    else:
        d = mesh.bounding_sphere.primitive.radius * 2
        if d > 0.8:
            print(f"    ⚠️  No scale.json found, mesh diameter={d:.3f}m (may be wrong scale)")
            print(f"       Run: conda activate depth-pro && python data/estimate_obj_scale.py")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    # Reuse pre-initialized models — do NOT re-create glctx/scorer/refiner
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug_dir=out_dir, debug=debug, glctx=glctx)

    # Use YcbineoatReader — already handles depth scale and K
    reader = YcbineoatReader(video_dir=scene_dir, shorter_side=None, zfar=np.inf)

    os.makedirs(os.path.join(out_dir, "ob_in_cam"),  exist_ok=True)
    os.makedirs(os.path.join(out_dir, "track_vis"),  exist_ok=True)

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            mask  = reader.get_mask(0).astype(bool)
            pose  = est.register(K=reader.K, rgb=color, depth=depth,
                                 ob_mask=mask, iteration=est_iter)
        else:
            pose  = est.track_one(rgb=color, depth=depth,
                                  K=reader.K, iteration=track_iter)

        np.savetxt(os.path.join(out_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt"),
                   pose.reshape(4, 4), fmt="%.6f")

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)
            imageio.imwrite(os.path.join(out_dir, "track_vis", f"{reader.id_strs[i]}.png"), vis)

        # Free intermediate tensors each frame
        torch.cuda.empty_cache()

    n = len(reader.color_files)
    print(f"    {n} frames → ob_in_cam + track_vis")
    return n


# ── 序列发现 ─────────────────────────────────────────────────────────────────
def discover_sequences(datasets):
    """
    Yield (dataset, seq_id, mesh_path) for sequences that have depth data.
    """
    for ds in datasets:
        ds_depth_dir = os.path.join(DEPTH_BASE, ds)
        if not os.path.isdir(ds_depth_dir): continue
        for seq_id in natsorted(os.listdir(ds_depth_dir)):
            if not os.path.exists(os.path.join(ds_depth_dir, seq_id, "depths.npz")):
                continue
            mesh_ds, obj_name = seq_to_obj(ds, seq_id)
            if obj_name is None: continue
            mesh_path = os.path.join(MESH_BASE, mesh_ds, obj_name, "mesh.ply")
            if not os.path.exists(mesh_path):
                continue
            yield ds, seq_id, mesh_ds, obj_name, mesh_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default=None,
                        choices=["arctic", "oakink", "ho3d_v3", "dexycb"])
    parser.add_argument("--seq",        default=None, help="seq_id 子串过滤")
    parser.add_argument("--limit",      type=int, default=0, help="最多处理几个序列（0=全部，smoke test用）")
    parser.add_argument("--start",      type=int, default=0, help="分片起始索引（含），多GPU并行用")
    parser.add_argument("--end",        type=int, default=0, help="分片结束索引（不含），0=处理到末尾")
    parser.add_argument("--redo",       action="store_true")
    parser.add_argument("--est-iter",   type=int, default=5)
    parser.add_argument("--track-iter", type=int, default=2)
    parser.add_argument("--keep-scene-dir", action="store_true",
                        help="don't delete the per-seq scratch dir under /tmp/fp_scenes "
                             "after FP finishes (default: delete to avoid filling /tmp)")
    parser.add_argument("--debug",      type=int, default=0,
                        help="0=无可视化 1=track_vis")
    args = parser.parse_args()

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3}GB")

    # Init FoundationPose models ONCE
    print("Initializing FoundationPose models...")
    import sys as _sys
    _sys.path.insert(0, FP_ROOT)
    scorer, refiner, glctx = init_fp_models()
    print("✅ Models ready\n")

    datasets = [args.dataset] if args.dataset else ["arctic", "oakink", "ho3d_v3", "dexycb"]

    seqs = list(discover_sequences(datasets))
    if args.seq:
        seqs = [(ds, sid, mds, onm, mp) for ds, sid, mds, onm, mp in seqs if args.seq in sid]
    if args.limit > 0:
        seqs = seqs[:args.limit]
    if args.start > 0 or args.end > 0:
        end = args.end if args.end > 0 else len(seqs)
        seqs = seqs[args.start:end]
        print(f"[Shard] seqs [{args.start}:{end}] = {len(seqs)}")

    print(f"\n{'='*60}")
    print(f" FoundationPose Batch Pose Estimation")
    print(f" Sequences : {len(seqs)}")
    print(f" Output    : {OUT_BASE}")
    print(f"{'='*60}\n")

    done = skipped = failed = 0

    for ds, seq_id, mesh_ds, obj_name, mesh_path in tqdm(seqs, desc="ObjPose"):
        out_dir = os.path.join(OUT_BASE, ds, seq_id)

        if not args.redo and os.path.isdir(os.path.join(out_dir, "ob_in_cam")):
            pose_files = glob(os.path.join(out_dir, "ob_in_cam", "*.txt"))
            if pose_files:
                tqdm.write(f"  ⏭  {ds}/{seq_id}: cached ({len(pose_files)} frames)")
                skipped += 1; continue

        tqdm.write(f"\n──────────────────────────────────────────────────────────")
        tqdm.write(f"▶ {ds}/{seq_id}  →  mesh: {mesh_ds}/{obj_name}")

        t0 = time.time()
        scene_dir = os.path.join(SCENE_TMP, ds, seq_id)
        os.makedirs(scene_dir, exist_ok=True)

        try:
            result = prepare_scene(ds, seq_id, mesh_ds, obj_name,
                                   scene_dir, shorter_side=SHORTER_SIDE)
            if result is None:
                tqdm.write(f"  ⚠️  prepare_scene failed"); failed += 1; continue
            n_frames, H, W = result
            tqdm.write(f"  Scene: {n_frames} frames @ {H}×{W}")

            n = run_fp(mesh_path, scene_dir, out_dir,
                       scorer=scorer, refiner=refiner, glctx=glctx,
                       est_iter=args.est_iter,
                       track_iter=args.track_iter,
                       debug=args.debug)
            tqdm.write(f"  ✅ {ds}/{seq_id}: {n} poses ({time.time()-t0:.1f}s)")
            done += 1

        except Exception as e:
            import traceback
            tqdm.write(f"  ❌ {e}")
            traceback.print_exc()
            failed += 1
        finally:
            import torch
            torch.cuda.empty_cache()   # 清显存，防止下一轮 OOM
            # leak fix: scene_dir is pure scratch (rgb/depth/masks copies of data already in
            # data_hub/); FP's actual outputs live in OUT_BASE. Without this, /tmp/fp_scenes/
            # accumulates ~54MB per seq until the disk fills (root cause of the 2026-05-13 outage).
            if not args.keep_scene_dir:
                shutil.rmtree(scene_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {OUT_BASE}")

    summary = {"done": done, "skipped": skipped, "failed": failed}
    os.makedirs(OUT_BASE, exist_ok=True)
    with open(os.path.join(OUT_BASE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

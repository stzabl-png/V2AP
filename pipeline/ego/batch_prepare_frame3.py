"""
batch_prepare_frame3.py — 自动对所有序列取第3帧 + 深度前景 mask

策略:
  - 固定取第 3 帧（index=2），此时手通常未入镜
  - 利用已有 Depth Pro 深度图，对中心区域最近的前景区域自动生成 mask
  - 已手动标注的序列（obj_recon_input/ 中已有 0.png）直接跳过

用法:
  conda activate depth-pro
  python data/batch_prepare_frame3.py                  # 全部
  python data/batch_prepare_frame3.py --dataset oakink --limit 5  # 测试
"""

import os, sys, argparse, numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

DEPTH_BASE     = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth")
RAW_BASE       = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
OUT_BASE       = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")
TACO_ALLOC_DIR = os.path.join(RAW_BASE, "taco", "Allocentric_RGB_Videos")

FRAME_IDX  = 2   # 第3帧（0-indexed）

# ── 与 batch_haptic.py 一致的发现函数 ────────────────────────────────────────
def discover_arctic(input_dir):
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "1")
            if not os.path.isdir(cam_dir): continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs: yield f"{subj}__{seq}", imgs

def discover_oakink(input_dir):
    for seq_id in natsorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_id)
        if not os.path.isdir(seq_dir): continue
        imgs = natsorted(glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
                         glob(os.path.join(seq_dir, "north_west_color_*.png")))
        if imgs: yield seq_id, imgs

def discover_ho3d(input_dir):
    for split in ["train", "evaluation"]:
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir): continue
        for seq_id in natsorted(os.listdir(split_dir)):
            rgb_dir = os.path.join(split_dir, seq_id, "rgb")
            if not os.path.isdir(rgb_dir): continue
            imgs = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
            if imgs: yield f"{split}__{seq_id}", imgs

def discover_dexycb(input_dir):
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for dt in natsorted(os.listdir(subj_dir)):
            dt_dir = os.path.join(subj_dir, dt)
            if not os.path.isdir(dt_dir): continue
            for serial in natsorted(os.listdir(dt_dir)):
                cam_dir = os.path.join(dt_dir, serial)
                if not os.path.isdir(cam_dir): continue
                imgs = natsorted(glob(os.path.join(cam_dir, "color_*.jpg")))
                if imgs: yield f"{subj}__{dt}__{serial}", imgs

def discover_taco_allocentric(input_dir):
    """
    Discover all (triplet, session, cam_serial) combos that have extracted jpg frames.
    seq_id = '{triplet}__{session}__{cam_serial}'  (safe for filesystem)
    Requires pre-extraction via:
        python tools/extract_taco_frames.py --mode pipeline --cam <serial>
    """
    base = TACO_ALLOC_DIR
    if not os.path.isdir(base):
        return
    for triplet in natsorted(os.listdir(base)):
        triplet_path = os.path.join(base, triplet)
        if not os.path.isdir(triplet_path):
            continue
        for session in natsorted(os.listdir(triplet_path)):
            session_path = os.path.join(triplet_path, session)
            if not os.path.isdir(session_path):
                continue
            for cam_serial in natsorted(os.listdir(session_path)):
                cam_dir = os.path.join(session_path, cam_serial)
                if not os.path.isdir(cam_dir):
                    continue
                imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")))
                if imgs:
                    # encode path as safe seq_id
                    safe_triplet = triplet.replace("/", "|")  # triplets can have spaces, keep ()s
                    seq_id = f"{safe_triplet}__{session}__{cam_serial}"
                    yield seq_id, imgs


DISCOVERERS = {
    "arctic":           (discover_arctic,           "arctic"),
    "oakink":           (discover_oakink,           "oakink_v1"),
    "ho3d_v3":          (discover_ho3d,             "ho3d_v3"),
    "dexycb":           (discover_dexycb,           "dexycb"),
    "taco_allocentric": (discover_taco_allocentric, "taco"),
}

# ── 深度前景 mask ─────────────────────────────────────────────────────────────
def depth_foreground_mask(depth, center_crop=0.6, percentile=40):
    """在中心裁剪区域内，选取最近 percentile% 的像素作为前景 mask。"""
    H, W = depth.shape
    # 中心区域
    r0, r1 = int(H * (1 - center_crop) / 2), int(H * (1 + center_crop) / 2)
    c0, c1 = int(W * (1 - center_crop) / 2), int(W * (1 + center_crop) / 2)

    center = depth[r0:r1, c0:c1]
    valid  = center[center > 0]
    if len(valid) == 0:
        # 全图备用
        valid = depth[depth > 0]
    if len(valid) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    threshold = np.percentile(valid, percentile)
    mask = ((depth > 0) & (depth <= threshold)).astype(np.uint8) * 255

    # 形态学清理
    try:
        import cv2
        k = np.ones((20, 20), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((10, 10), np.uint8))
    except ImportError:
        pass

    return mask


def get_depth_for_frame(depth_dir, frame_name):
    """从 depths.npz + frame_ids.txt 中找到对应帧的深度图。"""
    depth_npz  = os.path.join(depth_dir, "depths.npz")
    frame_file = os.path.join(depth_dir, "frame_ids.txt")
    if not os.path.exists(depth_npz) or not os.path.exists(frame_file):
        return None, None
    depths     = np.load(depth_npz)["depths"]
    frame_ids  = open(frame_file).read().strip().split("\n")
    fname      = os.path.basename(frame_name)
    if fname in frame_ids:
        idx = frame_ids.index(fname)
        return depths[idx], os.path.join(depth_dir, "K.txt")
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=list(DISCOVERERS.keys()))
    parser.add_argument("--limit",   type=int, default=0)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DISCOVERERS.keys())
    total_done = total_skip = total_fail = 0

    for ds in datasets:
        fn, folder = DISCOVERERS[ds]
        input_dir  = os.path.join(RAW_BASE, folder)
        depth_dir  = os.path.join(DEPTH_BASE, ds)
        out_dir_ds = os.path.join(OUT_BASE, ds)

        if not os.path.isdir(input_dir):
            print(f"⚠️  {ds}: raw dir not found, skip"); continue

        sequences = list(fn(input_dir))
        if args.limit > 0:
            sequences = sequences[:args.limit]

        print(f"\n{'─'*50}")
        print(f"▶ {ds}: {len(sequences)} sequences")
        done = skip = fail = 0

        for seq_id, img_paths in tqdm(sequences, desc=f"AutoMask/{ds}"):
            out_seq  = os.path.join(out_dir_ds, seq_id)
            img_out  = os.path.join(out_seq, "image.png")
            mask_out = os.path.join(out_seq, "0.png")

            # 已手动标注 → 跳过
            if os.path.exists(img_out) and os.path.exists(mask_out):
                skip += 1; continue

            try:
                # 取第 FRAME_IDX 帧（不足则取最后一帧）
                fi = min(FRAME_IDX, len(img_paths) - 1)
                frame_path = img_paths[fi]

                # 读取 RGB
                rgb = Image.open(frame_path).convert("RGB")
                W, H = rgb.size

                # 获取深度（优先使用已有 Depth Pro 结果）
                ds_depth_dir = os.path.join(depth_dir, seq_id)
                depth, k_path = get_depth_for_frame(ds_depth_dir, frame_path)

                if depth is not None:
                    mask = depth_foreground_mask(depth, center_crop=0.6, percentile=40)
                    K = np.loadtxt(k_path)
                else:
                    # 无深度：中心椭圆备用 mask
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cy, cx, ry, rx = H//2, W//2, int(H*0.3), int(W*0.3)
                    Y, X = np.ogrid[:H, :W]
                    mask[((Y-cy)/ry)**2 + ((X-cx)/rx)**2 <= 1] = 255
                    K = np.array([[max(H,W),0,W/2],[0,max(H,W),H/2],[0,0,1]])

                # 保存
                os.makedirs(out_seq, exist_ok=True)
                rgb.save(img_out)
                Image.fromarray(mask).save(mask_out)
                np.savetxt(os.path.join(out_seq, "K.txt"), K, fmt="%.6f")
                with open(os.path.join(out_seq, "source_frame.txt"), "w") as f:
                    f.write(f"{frame_path}\nframe_idx={fi}\nauto=True\n")

                done += 1

            except Exception as e:
                tqdm.write(f"  ❌ {seq_id}: {e}")
                fail += 1

        print(f"  ✅ {ds}: {done} auto  ⏭ {skip} manual/existing  ❌ {fail} failed")
        total_done += done; total_skip += skip; total_fail += fail

    print(f"\n{'='*60}")
    print(f"✅ 自动: {total_done}  ⏭ 已有手动标注: {total_skip}  ❌ 失败: {total_fail}")
    print(f"输出: {OUT_BASE}")
    print(f"\n上传云端:")
    print(f"  rsync -avz data_hub/ProcessedData/obj_recon_input/ sam3d-gpu:~/input/")


if __name__ == "__main__":
    main()

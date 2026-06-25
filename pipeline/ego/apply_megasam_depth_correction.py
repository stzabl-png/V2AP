#!/usr/bin/env python3
"""
apply_megasam_depth_correction.py — 给 MegaSAM ego 深度套一个固定全局校正系数。

背景:MegaSAM 单目 metric 深度在 egocentric 近距离(手持物体 ~0.8m)系统性高估
绝对尺度 ~1.9–2×(在 HOI4D 上用 objpose GT 距离实测,depthErr 中位 ~1.9)。
我们不逐序列用 GT(那不可泛化),而是烘焙一个固定常数 factor=0.5 进 depth.npz,
让深度逼近真实 metric。下游(scale 估计 / FoundationPose)直接读校正后的 depth.npz,
消费端代码无需改动。

幂等 & 可回退:
  - 首次运行把原始 MegaSAM 深度备份到 depth_megasam_raw.npz。
  - 之后**始终从 raw 备份 × factor** 重算,所以重复运行 / 改 factor 都不会叠乘。
  - meta.json 写入 depth_correction 字段标记。

用法:
  python pipeline/ego/apply_megasam_depth_correction.py                 # hoi4d, factor=0.5
  python pipeline/ego/apply_megasam_depth_correction.py --factor 0.52   # 实测最优
  python pipeline/ego/apply_megasam_depth_correction.py --restore       # 还原成原始 MegaSAM
"""
import os, sys, json, glob, argparse
import numpy as np

PROJ      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPTH_BASE = os.path.join(PROJ, "data_hub", "ProcessedData", "egocentric_depth")

RAW_NAME  = "depth_megasam_raw.npz"   # pristine backup
NPZ_NAME  = "depth.npz"
META_NAME = "meta.json"


def process_dir(depth_dir, factor, restore=False):
    npz_path  = os.path.join(depth_dir, NPZ_NAME)
    raw_path  = os.path.join(depth_dir, RAW_NAME)
    meta_path = os.path.join(depth_dir, META_NAME)
    if not os.path.exists(npz_path):
        print(f"  ⚠  no depth.npz in {depth_dir}")
        return False

    # meta (optional)
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}

    if restore:
        if not os.path.exists(raw_path):
            print(f"  ⏭  no raw backup, nothing to restore: {depth_dir}")
            return False
        raw = np.load(raw_path)["depths"].astype(np.float32)
        np.savez_compressed(npz_path, depths=raw)
        meta.pop("depth_correction", None)
        meta.pop("depth_correction_note", None)
        if meta:
            json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"  ↩  restored raw MegaSAM depth ({raw.shape})")
        return True

    # 确保有原始备份;之后总是从 raw 重算(避免叠乘)
    if not os.path.exists(raw_path):
        cur = np.load(npz_path)["depths"].astype(np.float32)
        np.savez_compressed(raw_path, depths=cur)
        base = cur
        print(f"  📦 backed up raw → {RAW_NAME}")
    else:
        base = np.load(raw_path)["depths"].astype(np.float32)

    corrected = (base * factor).astype(np.float32)
    np.savez_compressed(npz_path, depths=corrected)
    meta["depth_correction"] = factor
    meta["depth_correction_note"] = (
        "MegaSAM ego close-range absolute-depth overestimate (~1.9-2x); "
        "fixed global constant, no GT used at inference")
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"  ✅ depth ×{factor}  "
          f"median {np.median(base):.3f}→{np.median(corrected):.3f}m  ({base.shape})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hoi4d",
                    help="egocentric_depth/<dataset> 子目录(默认 hoi4d)")
    ap.add_argument("--factor", type=float, default=0.5,
                    help="深度乘的全局校正系数(默认 0.5)")
    ap.add_argument("--restore", action="store_true",
                    help="从 raw 备份还原原始 MegaSAM 深度")
    args = ap.parse_args()

    base_dir = os.path.join(DEPTH_BASE, args.dataset)
    dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))
    if not dirs:
        print(f"❌ no sequences under {base_dir}")
        sys.exit(1)

    action = "RESTORE" if args.restore else f"×{args.factor}"
    print(f"{'='*60}\n MegaSAM depth correction [{action}]  dataset={args.dataset}\n{'='*60}")
    n = 0
    for d in dirs:
        print(f"── {os.path.basename(d)}")
        if process_dir(d, args.factor, restore=args.restore):
            n += 1
    print(f"\n{'='*60}\n done: {n}/{len(dirs)} sequences\n{'='*60}")


if __name__ == "__main__":
    main()

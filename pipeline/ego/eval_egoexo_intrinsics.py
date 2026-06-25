#!/usr/bin/env python3
"""
eval_egoexo_intrinsics.py — 对比 Depth Pro / MegaSAM 估计内参 vs Ego-Exo4D GT

用法:
    # Exo 第三人称（用 Depth Pro 估内参）
    conda activate depth-pro
    python data/eval_egoexo_intrinsics.py --mode exo

    # Ego 第一人称（用 MegaSAM 估内参）
    cd mega-sam
    conda run -n mega_sam python ../data/eval_egoexo_intrinsics.py --mode ego

输出:
    每个 take 的 GT fx vs 估计 fx，以及误差百分比
"""

import os, sys, json, argparse, glob
import numpy as np
import cv2

EGO_DATA   = os.path.expanduser("~/ego_exo4d_data")
CAM_POSE_DIR = os.path.join(EGO_DATA, "annotations/ego_pose/train/camera_pose")


# ── GT 内参加载 ────────────────────────────────────────────────────────────────
# Build uid→filename map once
_UID_TO_FILE = {}
def _build_uid_map():
    if _UID_TO_FILE: return
    for f in glob.glob(os.path.join(CAM_POSE_DIR, "*.json")):
        try:
            with open(f) as fp:
                meta = json.load(fp)["metadata"]
            _UID_TO_FILE[meta["take_uid"]] = f
        except Exception:
            pass

def load_gt_intrinsics(take_uid):
    """从 camera_pose/*.json 加载 GT 内参"""
    _build_uid_map()
    path = _UID_TO_FILE.get(take_uid)
    if not path:
        return {}
    with open(path) as fp:
        d = json.load(fp)
    result = {}
    for cam_id, cam in d.items():
        if cam_id == "metadata": continue
        K = cam.get("camera_intrinsics")
        if K:
            result[cam_id] = {
                "fx": K[0][0], "fy": K[1][1],
                "cx": K[0][2], "cy": K[1][2],
            }
    return result


# ── 从视频抽帧 ────────────────────────────────────────────────────────────────
def extract_frame(video_path, frame_idx=30):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total // 2))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ── Depth Pro 估计内参（exo 相机）────────────────────────────────────────────
def estimate_fx_depthpro(img_path):
    """Run Depth Pro and return estimated fx in pixels"""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "../third_party/ml-depth-pro/src"))
    import depth_pro

    if not hasattr(estimate_fx_depthpro, "_model"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, transform = depth_pro.create_model_and_transforms(
            device=device, precision=torch.float16)
        model.eval()
        estimate_fx_depthpro._model = (model, transform, device)

    model, transform, device = estimate_fx_depthpro._model
    image, _, f_px = depth_pro.load_rgb(img_path)
    with torch.no_grad():
        pred = model.infer(transform(image).unsqueeze(0).to(device), f_px=f_px)
    H, W = pred["depth"].shape[-2:]
    return float(pred["focallength_px"]), W, H


# ── UniDepth 估计内参（Ego/Aria 相机）────────────────────────────────────
def estimate_fx_unidepth(img):
    """Run UniDepth on a single frame (numpy BGR) and return estimated fx.
    Mirrors the run_unidepth_on_dir logic in batch_megasam.py.
    """
    import torch, sys as _sys
    import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as _cfg
_mega = _cfg.MEGASAM_DIR
    for _p in [_mega, f"{_mega}/UniDepth", f"{_mega}/unidepth"]:
        if _p not in _sys.path: _sys.path.insert(0, _p)

    if not hasattr(estimate_fx_unidepth, "_model"):
        from unidepth.models import UniDepthV2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = UniDepthV2.from_pretrained(
            "lpiccinelli/unidepth-v2-vitl14",
            revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
        model  = model.to(device).eval()
        estimate_fx_unidepth._model  = model
        estimate_fx_unidepth._device = device

    model  = estimate_fx_unidepth._model
    device = estimate_fx_unidepth._device
    import torch
    import numpy as np

    # BGR→RGB, HWC→CHW, [0,1]
    rgb = img[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model.infer(tensor)

    K = preds["intrinsics"].squeeze().cpu().numpy()  # (3,3)
    fx = float(K[0, 0])
    H, W = img.shape[:2]
    return fx, W, H


# ── 主评估循环 ────────────────────────────────────────────────────────────────
def eval_exo(takes_json_path, uid_list):
    """Depth Pro on exo cameras"""
    with open(takes_json_path) as f:
        takes = json.load(f)
    take_map = {t["take_uid"]: t for t in takes}

    results = []
    print(f"\n{'Take':<35} {'GT fx':>8} {'Est fx':>8} {'误差':>8} {'分辨率'}")
    print("-" * 75)

    for uid in uid_list:
        t = take_map.get(uid)
        if not t: continue
        root = os.path.join(EGO_DATA, t["root_dir"])
        gt = load_gt_intrinsics(uid)

        # 找 best_exo 相机的视频（downscaled/448 路径）
        best_cam = t.get("best_exo", "cam01")
        take_name = t["take_name"]
        vid_dir = os.path.join(EGO_DATA, "takes", take_name,
                               "frame_aligned_videos", "downscaled", "448")
        mp4 = os.path.join(vid_dir, f"{best_cam}.mp4")
        # fallback: try cam01~cam05
        if not os.path.exists(mp4):
            for c in ["cam01","cam02","cam03","cam04","cam05"]:
                p = os.path.join(vid_dir, f"{c}.mp4")
                if os.path.exists(p):
                    mp4 = p; best_cam = c; break
            else:
                mp4 = None

        if mp4 is None:
            print(f"  ⚠️  {t['take_name']:<33} 视频未找到 (已下载?)")
            continue

        # 抽帧保存
        frame = extract_frame(mp4, frame_idx=60)
        if frame is None: continue
        img_path = f"/tmp/dp_eval_{uid[:8]}.jpg"
        cv2.imwrite(img_path, frame)

        try:
            est_fx, W, H = estimate_fx_depthpro(img_path)
            gt_fx = gt.get(best_cam, {}).get("fx", None)
            if gt_fx:
                err = (est_fx - gt_fx) / gt_fx * 100
                # Depth Pro 在降采样图上估计，需按分辨率缩放到 GT 分辨率
                gt_W = gt.get(best_cam, {}).get("cx", W/2) * 2
                scale = gt_W / W
                est_fx_scaled = est_fx * scale
                err_scaled = (est_fx_scaled - gt_fx) / gt_fx * 100
                print(f"  {t['take_name']:<33} {gt_fx:>8.0f} {est_fx_scaled:>8.0f}"
                      f" {err_scaled:>+7.1f}%  {W}×{H}→GT {int(gt_W)}px")
                results.append({"take": t["take_name"], "gt_fx": gt_fx,
                                 "est_fx": est_fx_scaled, "err": err_scaled})
            else:
                print(f"  {t['take_name']:<33} GT 不存在")
        except Exception as e:
            print(f"  ❌ {t['take_name']}: {e}")

    if results:
        errs = [abs(r["err"]) for r in results]
        print(f"\n{'='*75}")
        print(f"  平均绝对误差: {np.mean(errs):.1f}%")
        print(f"  最大误差:     {max(errs):.1f}%")
        print(f"  误差 <5% 的:  {sum(1 for e in errs if e<5)}/{len(errs)}")
    return results


def eval_ego(takes_json_path, uid_list):
    """MegaSAM on ego (aria) cameras — aria GT fx is always 150px @ 511×511"""
    with open(takes_json_path) as f:
        takes = json.load(f)
    take_map = {t["take_uid"]: t for t in takes}

    ARIA_GT_FX = 150.0  # fixed factory calibration
    ARIA_RES   = 511

    results = []
    print(f"\n{'Take':<35} {'GT fx':>8} {'Est fx':>8} {'误差':>8}")
    print("-" * 60)

    for uid in uid_list:
        t = take_map.get(uid)
        if not t: continue
        root = os.path.join(EGO_DATA, t["root_dir"])
        take_name = t["take_name"]
        vid_dir = os.path.join(EGO_DATA, "takes", take_name,
                               "frame_aligned_videos", "downscaled", "448")
        # Aria RGB stream: aria01_214-1.mp4
        mp4 = os.path.join(vid_dir, "aria01_214-1.mp4")
        if not os.path.exists(mp4):
            # fallback: any aria mp4
            aria_vids = glob.glob(os.path.join(vid_dir, "aria01_21*.mp4"))
            mp4 = aria_vids[0] if aria_vids else None

        if mp4 is None:
            print(f"  ⚠️  {t['take_name']:<33} ego 视频未找到")
            continue

        out_dir = f"/tmp/megasam_eval/{uid[:8]}"
        frame = extract_frame(mp4, frame_idx=60)
        if frame is None:
            print(f"  ⚠️  {t['take_name']:<33} 抽帧失败")
            continue

        try:
            est_fx, W, H = estimate_fx_unidepth(frame)
            err = (est_fx - ARIA_GT_FX) / ARIA_GT_FX * 100
            print(f"  {t['take_name']:<33} {ARIA_GT_FX:>8.1f} {est_fx:>8.1f} {err:>+7.1f}%  ({W}×{H})")
            results.append({"take": t["take_name"], "gt_fx": ARIA_GT_FX,
                             "est_fx": est_fx, "err": err})
        except Exception as e:
            print(f"  ❌ {t['take_name']}: {e}")

    if results:
        errs = [abs(r["err"]) for r in results]
        print(f"\n  平均绝对误差: {np.mean(errs):.1f}%")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Eval Depth Pro / MegaSAM intrinsics on Ego-Exo4D")
    p.add_argument("--mode", choices=["exo", "ego"], required=True)
    p.add_argument("--uids_file", default="/tmp/egoexo_sample10_uids.txt")
    args = p.parse_args()

    with open(args.uids_file) as f:
        uids = f.read().strip().split()
    print(f"评估 {len(uids)} 个 takes，模式: {args.mode}")

    takes_json = os.path.join(EGO_DATA, "takes.json")

    if args.mode == "exo":
        print("\n=== Depth Pro 内参估计精度 (Exo 第三人称) ===")
        eval_exo(takes_json, uids)
    else:
        print("\n=== MegaSAM 内参估计精度 (Ego 第一人称/Aria) ===")
        eval_ego(takes_json, uids)


if __name__ == "__main__":
    main()

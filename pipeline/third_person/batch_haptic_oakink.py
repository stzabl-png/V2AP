#!/usr/bin/env python3
"""
batch_haptic_oakink.py — OakInk 序列 HaPTIC 批量推理 (Step 1)
=============================================================
从 VLM 关键帧标注中获取 MANO 帧范围，对 OakInk north_west 摄像机
图像运行 HaPTIC，保存 MANO 手部顶点缓存。

与 batch_haptic_arctic.py 的核心区别：
  - 图像路径: OakInk stream_release_v2 而非 ARCTIC cropped_images
  - 内参: OakInk GT anno/cam_intr/ 而非 MegaSAM 估计 (更准!)
  - 帧范围: oakink_keyframe_annot.json

用法:
  conda activate haptic
  cd $PROJ
  python data/batch_haptic_oakink.py                          # 全部
  python data/batch_haptic_oakink.py --seq A01001_0001_0000  # 单序列
  python data/batch_haptic_oakink.py --cam north_west        # 指定摄像机
"""

import os, sys, glob, json, pickle, argparse, shutil, tempfile
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
sys.path.insert(0, config.HAPTIC_DIR)

# ── 路径配置 ─────────────────────────────────────────────────────────────────
OAKINK_STREAM  = os.path.join(config.DATA_HUB, 'RawData', 'ThirdPersonRawData', 'oakink_v1')
OAKINK_CAM_K   = os.path.join(config.OAKINK_ANNO_DIR, 'image', 'anno', 'cam_intr') if config.OAKINK_ANNO_DIR else ''
ANNOT_JSON     = os.path.join(config.OUTPUT_DIR, 'oakink_keyframe_annot.json')
CACHE_DIR      = os.path.join(config.OUTPUT_DIR, 'haptic_oakink_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# OakInk 摄像机名称 → cam_idx 映射
# 0=north_west 1=south_west 2=north_east 3=south_east (可能需要验证)
CAM_IDX_MAP = {
    'north_west':  0,
    'south_west':  1,
    'north_east':  2,
    'south_east':  3,
}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ── 内参加载 ─────────────────────────────────────────────────────────────────
def load_oakink_K(seq_id, cam_name, fallback_frame=None):
    """从 OakInk anno/cam_intr/ 加载 GT 3x3 内参矩阵。
    
    文件命名: {seq}__{timestamp}__0__{frame}__{cam_idx}.pkl
    策略: 查找该序列任意一帧的对应 cam_idx 文件，取中位数(若多帧)。
    """
    cam_idx = CAM_IDX_MAP.get(cam_name, 0)
    pattern = os.path.join(OAKINK_CAM_K, f'{seq_id}__*__{cam_idx}.pkl')
    files = sorted(glob.glob(pattern))
    
    if not files:
        # 备选：尝试所有 cam_idx 文件（按序列前缀匹配）
        pattern_any = os.path.join(OAKINK_CAM_K, f'{seq_id}__*.pkl')
        all_files = sorted(glob.glob(pattern_any))
        # 筛选 cam_idx 结尾的文件
        files = [f for f in all_files if f.endswith(f'__{cam_idx}.pkl')]
    
    if not files:
        return None  # 找不到内参，HaPTIC 会用默认焦距

    Ks = []
    # 最多取 10 帧做平均（内参通常不变）
    for f in files[:10]:
        try:
            K = pickle.load(open(f, 'rb'))
            if isinstance(K, np.ndarray) and K.shape == (3, 3):
                Ks.append(K)
        except Exception:
            pass
    
    if not Ks:
        return None
    return np.median(np.stack(Ks, axis=0), axis=0)  # (3,3)


# ── 图像帧加载 ────────────────────────────────────────────────────────────────
def get_frame_paths(seq_id, cam_name, frame_start, frame_end):
    """获取 [frame_start, frame_end] 帧范围内的图像路径列表。"""
    seq_dir = os.path.join(OAKINK_STREAM, seq_id)
    if not os.path.isdir(seq_dir):
        return []
    
    # OakInk 目录结构: {seq}/{timestamp}/{cam}_color_{frame}.png
    timestamps = sorted(os.listdir(seq_dir))
    if not timestamps:
        return []
    ts = timestamps[0]  # 通常只有一个 timestamp 子目录
    
    img_paths = []
    for fi in range(frame_start, frame_end + 1):
        p = os.path.join(seq_dir, ts, f'{cam_name}_color_{fi}.png')
        if os.path.exists(p):
            img_paths.append((fi, p))
    
    return img_paths  # [(frame_idx, path), ...]


# ── HaPTIC 模型加载 ───────────────────────────────────────────────────────────
def load_haptic_model():
    from omegaconf import OmegaConf
    import haptic.models.haptic as haptic_module

    ckpt     = os.path.join(config.HAPTIC_DIR, 'output/release/mix_all/checkpoints/last.ckpt')
    cfg_path = os.path.join(config.HAPTIC_DIR, 'output/release/mix_all/config.yaml')

    old_cwd = os.getcwd()
    os.chdir(config.HAPTIC_DIR)

    cfg    = OmegaConf.load(cfg_path)
    if 'PRETRAINED_WEIGHTS' in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
    class_ = getattr(haptic_module, cfg.MODEL.get('TARGET', 'HAPTIC'))
    model  = class_(cfg)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict'], strict=False)
    model  = model.to(device)
    model.eval()

    os.chdir(old_cwd)
    return model, cfg


# ── HaPTIC 推理（带 GT 内参）────────────────────────────────────────────────
def run_haptic_oakink(model, model_cfg, frame_paths, K):
    """对 OakInk 图像帧运行 HaPTIC，返回 {frame_idx → (778,3) verts}。
    
    K: 3x3 GT 内参矩阵 (可以为 None，HaPTIC 会用默认焦距估计)
    frame_paths: [(orig_frame_idx, img_path), ...]
    """
    from haptic.datasets.seq2clip import split_to_list_dl
    from nnutils.hand_utils import ManopthWrapper
    from nnutils import model_utils, geom_utils, mesh_utils
    from nnutils.det_utils import parse_det_seq
    from haptic.utils.renderer import cam_crop_to_full_w_depth, cam_crop_to_full_w_pp
    from data.megasam_utils import K_as_haptic_intrinsics

    hand_wrapper   = ManopthWrapper(mano_path=config.HAPTIC_MANO_DIR).to(device)
    overlap = 1 if model_cfg.MODEL.NUM_FRAMES > 1 else 0

    # GT 内参转换为 HaPTIC 9-element format
    haptic_intrinsics = K_as_haptic_intrinsics(K) if K is not None else None
    if haptic_intrinsics:
        print(f'  [OakInk-K] fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
              f'cx={K[0,2]:.1f} cy={K[1,2]:.1f}')

    # 建立临时目录，把帧软链接为连续数字文件名（HaPTIC 要求）
    tmp_dir = tempfile.mkdtemp(prefix='haptic_oakink_')
    frame_idx_map = {}  # local_idx → orig_frame_idx

    try:
        for local_i, (orig_fi, path) in enumerate(frame_paths):
            ext = os.path.splitext(path)[1]
            dst = os.path.join(tmp_dir, f'{local_i:06d}{ext}')
            os.symlink(os.path.abspath(path), dst)
            frame_idx_map[local_i] = orig_fi

        class DataCfg:
            video_dir  = tmp_dir
            video_list = None
        class DemoCfg:
            box_mode = 'box_size_same'

        seq_list = parse_det_seq(DataCfg(), DemoCfg(), skip=False,
                                 intrinsics=haptic_intrinsics)
        if not seq_list:
            return None

        all_verts = {}
        for seq in seq_list:
            seq_dl = split_to_list_dl(
                model_cfg, seq, model_cfg.MODEL.NUM_FRAMES, overlap,
                box_mode='box_size_same', load_depth=False, rescale_factor=2)

            depth0 = 0
            for b, bs in enumerate(seq_dl):
                bs = model_utils.to_cuda(bs, device)
                with torch.no_grad():
                    pred = model(bs)

                if b == 0:
                    W, H = bs['img_size'][0, 0:1].split([1,1], -1)
                    W, H = W.squeeze(-1), H.squeeze(-1)
                    cam_full = cam_crop_to_full_w_pp(
                        pred['pred_cam'][0:1], bs['intr'][0, 0:1],
                        H, W, bs['box_center'][0, 0:1], bs['box_size'][0, 0:1])
                    depth0 = cam_full[..., 2]

                offset = pred['pred_depth'][0:1]
                pred['pred_depth'] = pred['pred_depth'] - offset + depth0
                depth0 = pred['pred_depth'][-1:]

                pHand, _ = hand_wrapper(
                    None,
                    geom_utils.matrix_to_axis_angle(pred['pred_mano_params']['hand_pose']).reshape(-1, 45),
                    geom_utils.matrix_to_axis_angle(pred['pred_mano_params']['global_orient']).reshape(-1, 3),
                    th_betas=pred['pred_mano_params']['betas'])

                W, H = bs['img_size'][0].split([1,1], -1)
                W, H = W.squeeze(-1), H.squeeze(-1)
                box_center = bs['box_center'][0].clone()
                flip = not bs['right'][0]
                if flip:
                    box_center[..., 0] = W - box_center[..., 0]

                cam_full = cam_crop_to_full_w_depth(
                    pred['pred_cam'], bs['intr'][0], H, W,
                    box_center, bs['box_size'][0], pred['pred_depth'].squeeze(1))

                cTp   = geom_utils.axis_angle_t_to_matrix(torch.zeros_like(cam_full), cam_full)
                cHand = mesh_utils.apply_transform(pHand, cTp)
                verts = cHand.verts_padded()
                if flip:
                    verts[..., 0] = -verts[..., 0]

                t0    = 0 if b == 0 else overlap
                names = [os.path.basename(e[0]) for e in bs['name']]
                for t in range(t0, len(names)):
                    local_idx = int(os.path.splitext(names[t])[0])
                    orig_fi   = frame_idx_map.get(local_idx, local_idx)
                    all_verts[orig_fi] = verts[t].cpu().numpy()  # (778,3)

        return all_verts  # {orig_frame_idx → (778,3)}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq',    default=None, help='单序列 ID (调试用)')
    ap.add_argument('--cam',    default='north_west', choices=list(CAM_IDX_MAP))
    ap.add_argument('--force',  action='store_true', help='覆盖已有缓存')
    args = ap.parse_args()

    # 加载 VLM 关键帧标注
    annot = json.load(open(ANNOT_JSON))
    print(f'📝 VLM 标注: {len(annot)} 个序列')

    seq_ids = [args.seq] if args.seq else sorted(annot.keys())
    print(f'待处理: {len(seq_ids)} 个  cam={args.cam}\n')

    print('Loading HaPTIC model...')
    model, model_cfg = load_haptic_model()
    print('✅ Model loaded\n')

    done = skipped = failed = no_hand = 0

    for seq_id in tqdm(seq_ids, desc='OakInk HaPTIC'):
        if seq_id not in annot:
            tqdm.write(f'  ⚠️ {seq_id}: 无标注，跳过')
            continue

        ann = annot[seq_id]
        mano_start = ann['mano_start']
        mano_end   = ann['mano_end']
        obj_id     = ann.get('obj_id', seq_id.rsplit('_', 2)[0])

        # 缓存路径: {seq_id}_{cam}.npz
        cache_name = f'{seq_id}_{args.cam}.npz'
        cache_path = os.path.join(CACHE_DIR, cache_name)

        if os.path.exists(cache_path) and not args.force:
            skipped += 1
            continue

        # 获取图像帧
        frame_paths = get_frame_paths(seq_id, args.cam, mano_start, mano_end)
        if len(frame_paths) < 2:
            tqdm.write(f'  ⚠️ {seq_id}: 图像不足 ({len(frame_paths)} 帧)')
            failed += 1
            continue

        # 加载 GT 内参
        K = load_oakink_K(seq_id, args.cam)
        if K is None:
            tqdm.write(f'  ⚠️ {seq_id}: 无 GT 内参，使用 HaPTIC 默认焦距')

        # 运行 HaPTIC
        try:
            verts_dict = run_haptic_oakink(model, model_cfg, frame_paths, K)
        except Exception as e:
            tqdm.write(f'  ✗ {seq_id}: {e}')
            failed += 1
            torch.cuda.empty_cache()
            continue

        if not verts_dict:
            tqdm.write(f'  ⚠️ {seq_id}: 未检测到手')
            no_hand += 1
            continue

        # 保存缓存
        np.savez_compressed(cache_path,
                            verts_dict=verts_dict,
                            seq_id=seq_id,
                            obj_id=obj_id,
                            cam=args.cam,
                            mano_start=mano_start,
                            mano_end=mano_end)
        done += 1
        tqdm.write(f'  ✅ {seq_id}: {len(verts_dict)} 帧手部顶点')
        torch.cuda.empty_cache()

    print(f'\n{"="*60}')
    print(f'  完成:{done}  跳过:{skipped}  失败:{failed}  无手:{no_hand}')
    print(f'  缓存: {CACHE_DIR}')


if __name__ == '__main__':
    os.chdir(config.HAPTIC_DIR)
    main()

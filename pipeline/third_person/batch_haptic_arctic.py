"""
Run HaPTIC on ARCTIC third-person cameras to compare with HaWoR egocentric.
Uses manually annotated grasp frame ranges.

Usage:
  conda activate haptic
  python -m data.batch_haptic_arctic

Evaluates contact IoU against GT MANO + GT mesh.
"""

import os, sys, json, shutil, tempfile, numpy as np, torch
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Project config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.megasam_utils import load_megasam_K, K_as_haptic_intrinsics

sys.path.insert(0, config.HAPTIC_DIR)

ARCTIC_ROOT = config.ARCTIC_ROOT
CROPPED_DIR = os.path.join(ARCTIC_ROOT, "arctic_data/data/cropped_images")
# Auto-detected grasp onset for ~300 sequences (clip_start/clip_end fields)
GRASP_ONSET_JSON = os.path.join(config.OUTPUT_DIR, "arctic_grasp_onset.json")
CACHE_DIR = config.HAPTIC_CACHE
RESULT_JSON = os.path.join(config.OUTPUT_DIR, "haptic_arctic_iou_results.json")

os.makedirs(CACHE_DIR, exist_ok=True)

CONTACT_TH = config.GT_CONTACT_TH
THIRD_PERSON_CAMS = [1, 2, 3, 4, 5, 6, 7, 8]  # all third-person cameras
device = "cuda:0"

OBJECTS = ['box','capsulemachine','espressomachine','ketchup','laptop',
           'microwave','mixer','notebook','phone','scissors','waffleiron']

def get_obj(seq):
    for o in sorted(OBJECTS, key=len, reverse=True):
        if seq.startswith(o): return o
    return None


def load_haptic_model():
    """Load HaPTIC model."""
    from omegaconf import OmegaConf
    import haptic.models.haptic as haptic_module
    
    ckpt = os.path.join(config.HAPTIC_DIR, "output/release/mix_all/checkpoints/last.ckpt")
    cfg_path = os.path.join(config.HAPTIC_DIR, "output/release/mix_all/config.yaml")
    
    old_cwd = os.getcwd()
    os.chdir(config.HAPTIC_DIR)
    
    cfg = OmegaConf.load(cfg_path)
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    class_ = getattr(haptic_module, cfg.MODEL.get("TARGET", "HAPTIC"))
    model = class_(cfg)
    model.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    
    os.chdir(old_cwd)
    return model, cfg


def run_haptic_on_images(model, model_cfg, img_paths, scene_name=None,
                         precomputed_K=None):
    """Run HaPTIC on a folder of images, return dict of frame_idx -> (778,3) verts.

    Args:
        model: loaded HaPTIC model.
        model_cfg: HaPTIC OmegaConf config.
        img_paths: list of absolute image paths.
        scene_name: optional sequence name used to look up MegaSAM intrinsics.
            When provided and MegaSAM output exists, the estimated K is passed
            to HaPTIC's ``parse_det_seq`` instead of the default heuristic
            (sqrt(W²+H²) focal + cx=W/2, cy=H/2).
        precomputed_K: optional 9-element list (row-major 3×3 K) to pass directly
            to HaPTIC, bypassing both MegaSAM lookup and the heuristic.
            Depth Pro K should be passed here for metric consistency.
    """
    from haptic.datasets.seq2clip import split_to_list_dl
    from nnutils.hand_utils import ManopthWrapper
    from nnutils import model_utils, geom_utils, mesh_utils
    from nnutils.det_utils import parse_det_seq
    from haptic.utils.renderer import cam_crop_to_full_w_depth, cam_crop_to_full_w_pp

    hand_wrapper = ManopthWrapper(mano_path=config.HAPTIC_MANO_DIR).to(device)
    overlap = 1 if model_cfg.MODEL.NUM_FRAMES > 1 else 0

    # --- Intrinsics priority: precomputed_K > MegaSAM > HaPTIC heuristic ------
    if precomputed_K is not None:
        haptic_intrinsics = precomputed_K   # already 9-element list
        print(f"  [K] Using precomputed intrinsics (Depth Pro)")
    else:
        megasam_K = load_megasam_K(scene_name) if scene_name else None
        if megasam_K is not None:
            haptic_intrinsics = K_as_haptic_intrinsics(megasam_K)
            print(f"  [MegaSAM] HaPTIC using K=[[{megasam_K[0,0]:.1f},0,{megasam_K[0,2]:.1f}],"
                  f"[0,{megasam_K[1,1]:.1f},{megasam_K[1,2]:.1f}],[0,0,1]] for {scene_name}")
        else:
            haptic_intrinsics = None  # HaPTIC uses its sqrt(W²+H²) heuristic
            if scene_name:
                print(f"  [MegaSAM] No intrinsics for {scene_name}, HaPTIC using default focal")
    # --------------------------------------------------------------------------
    
    tmp_dir = tempfile.mkdtemp(prefix="haptic_arctic_")
    try:
        for i, p in enumerate(img_paths):
            ext = os.path.splitext(p)[1]
            os.symlink(os.path.abspath(p), os.path.join(tmp_dir, f"{i:06d}{ext}"))
        
        class DataCfg:
            video_dir = tmp_dir
            video_list = None
        class DemoCfg:
            box_mode = "box_size_same"
        
        seq_list = parse_det_seq(DataCfg(), DemoCfg(), skip=False,
                                  intrinsics=haptic_intrinsics)
        if not seq_list:
            return None
        
        all_verts = {}
        for seq in seq_list:
            seq_dl = split_to_list_dl(
                model_cfg, seq, model_cfg.MODEL.NUM_FRAMES, overlap,
                box_mode="box_size_same", load_depth=False, rescale_factor=2)
            
            depth0 = 0
            for b, bs in enumerate(seq_dl):
                bs = model_utils.to_cuda(bs, device)
                with torch.no_grad():
                    pred = model(bs)
                
                if b == 0:
                    W, H = bs["img_size"][0, 0:1].split([1, 1], -1)
                    W, H = W.squeeze(-1), H.squeeze(-1)
                    cam_full = cam_crop_to_full_w_pp(
                        pred["pred_cam"][0:1], bs["intr"][0, 0:1],
                        H, W, bs["box_center"][0, 0:1], bs["box_size"][0, 0:1])
                    depth0 = cam_full[..., 2]
                
                offset = pred["pred_depth"][0:1]
                pred["pred_depth"] = pred["pred_depth"] - offset + depth0
                depth0 = pred["pred_depth"][-1:]
                
                pHand, _ = hand_wrapper(
                    None,
                    geom_utils.matrix_to_axis_angle(pred["pred_mano_params"]["hand_pose"]).reshape(-1, 45),
                    geom_utils.matrix_to_axis_angle(pred["pred_mano_params"]["global_orient"]).reshape(-1, 3),
                    th_betas=pred["pred_mano_params"]["betas"])
                
                W, H = bs["img_size"][0].split([1, 1], -1)
                W, H = W.squeeze(-1), H.squeeze(-1)
                box_center = bs["box_center"][0].clone()
                flip = not bs["right"][0]
                if flip:
                    box_center[..., 0] = W - box_center[..., 0]
                
                cam_full = cam_crop_to_full_w_depth(
                    pred["pred_cam"], bs["intr"][0], H, W,
                    box_center, bs["box_size"][0], pred["pred_depth"].squeeze(1))
                
                cTp = geom_utils.axis_angle_t_to_matrix(torch.zeros_like(cam_full), cam_full)
                cHand = mesh_utils.apply_transform(pHand, cTp)
                
                verts = cHand.verts_padded()
                if flip:
                    verts[..., 0] = -verts[..., 0]
                
                t0 = 0 if b == 0 else overlap
                names = [os.path.basename(e[0]) for e in bs["name"]]
                for t in range(t0, len(names)):
                    local_idx = int(os.path.splitext(names[t])[0])
                    all_verts[local_idx] = verts[t].cpu().numpy()
        
        return all_verts
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_contact_haptic(subj, seq, start, end, verts_dict, cam_id):
    """Evaluate HaPTIC contact IoU against GT."""
    obj_name = get_obj(seq)
    import trimesh
    
    mg = np.load(f'{ARCTIC_ROOT}/raw_seqs/{subj}/{seq}.mano.npy', allow_pickle=True).item()
    od = np.load(f'{ARCTIC_ROOT}/raw_seqs/{subj}/{seq}.object.npy', allow_pickle=True)
    
    # Load camera extrinsics from ARCTIC misc.json
    # world2cam is a list of 8 4x4 matrices per subject (index 0-7 for cam 0-7)
    misc_path = f'{ARCTIC_ROOT}/meta/misc.json'
    with open(misc_path) as f:
        misc = json.load(f)
    
    w2c_4x4 = np.array(misc[subj]['world2cam'][cam_id])  # 4x4
    R_cam = w2c_4x4[:3, :3]  # 3x3
    T_cam = w2c_4x4[:3, 3]   # 3
    
    m = trimesh.load(f'{ARCTIC_ROOT}/meta/object_vtemplates/{obj_name}/mesh.obj', process=False)
    np.random.seed(42)
    opt = np.array(m.sample(2048)) / 1000.0
    
    N_gt = od.shape[0]
    N_clip = end - start + 1
    
    # GT MANO using smplx
    import smplx
    mano_r = smplx.create(config.MANO_MODELS, 
                          model_type='mano', is_rhand=True, use_pca=False, flat_hand_mean=False)
    
    sl = slice(start, min(end + 1, N_gt))
    N_sl = min(end + 1, N_gt) - start
    
    out_r = mano_r(
        global_orient=torch.from_numpy(mg['right']['rot'][sl]).float(),
        hand_pose=torch.from_numpy(mg['right']['pose'][sl]).float(),
        betas=torch.from_numpy(np.tile(mg['right']['shape'],(N_sl,1))).float(),
        transl=torch.from_numpy(mg['right']['trans'][sl]).float())
    gt_r = out_r.vertices.detach().numpy()  # (N_sl, 778, 3)
    
    ious = []
    for i in range(0, min(len(verts_dict), N_sl), 5):
        if i not in verts_dict: continue
        fi = start + i
        if fi >= N_gt: break
        
        # Object in cam space
        oR = Rotation.from_rotvec(od[fi,1:4]).as_matrix()
        ot = od[fi,4:7]/1000.0
        ow = (oR@opt.T).T+ot
        oc = (R_cam@ow.T).T+T_cam
        
        # GT hand in cam space
        gc = (R_cam@gt_r[i].T).T+T_cam
        gd,_ = cKDTree(gc).query(oc)
        gt_con = (gd < CONTACT_TH).astype(int)
        if gt_con.sum() == 0: continue
        
        # HaPTIC hand (already in camera space from the model)
        hr = verts_dict[i]
        # Centroid align (same as HaWoR eval)
        off = gc.mean(0) - hr.mean(0)
        hr_a = hr + off
        hd,_ = cKDTree(hr_a).query(oc)
        hw_con = (hd < CONTACT_TH).astype(int)
        
        inter = ((gt_con>0)&(hw_con>0)).sum()
        union = ((gt_con>0)|(hw_con>0)).sum()
        ious.append(inter/union if union>0 else 0)
    
    return ious


def main():
    os.chdir(config.HAPTIC_DIR)
    grasp_onset = json.load(open(GRASP_ONSET_JSON))
    print(f"📝 Grasp onset sequences: {len(grasp_onset)}")
    
    print("Loading HaPTIC model...")
    model, model_cfg = load_haptic_model()
    print("✅ Model loaded")
    
    # Use cam 1 as default third-person camera
    CAM_ID = 1
    
    results = {}
    all_ious = []
    
    for key, ann in tqdm(sorted(grasp_onset.items()), desc="Processing"):
        subj, seq = ann['subject'], ann['sequence']
        start, end = ann['clip_start'], ann['clip_end']
        
        cache_path = os.path.join(CACHE_DIR, f'{key}_cam{CAM_ID}.npz')
        
        try:
            if os.path.exists(cache_path):
                # Load cached
                cd = dict(np.load(cache_path, allow_pickle=True))
                verts_dict = cd['verts_dict'].item()
            else:
                # Get image paths for this camera
                cam_dir = os.path.join(CROPPED_DIR, subj, seq, str(CAM_ID))
                if not os.path.isdir(cam_dir):
                    tqdm.write(f"  {subj}/{seq}: cam {CAM_ID} not found")
                    continue
                
                frames = sorted(os.listdir(cam_dir))
                img_paths = [os.path.join(cam_dir, frames[i]) 
                            for i in range(start, min(end+1, len(frames)))]
                
                if len(img_paths) < 2:
                    tqdm.write(f"  {subj}/{seq}: too few images")
                    continue
                
                # Run HaPTIC
                verts_dict = run_haptic_on_images(
                    model, model_cfg, img_paths,
                    scene_name=seq,  # enables MegaSAM K lookup
                )
                
                if verts_dict is None or len(verts_dict) == 0:
                    tqdm.write(f"  {subj}/{seq}: no hand detected")
                    continue
                
                # Cache
                np.savez_compressed(cache_path, verts_dict=verts_dict)
            
            # Evaluate
            ious = evaluate_contact_haptic(subj, seq, start, end, verts_dict, CAM_ID)
            
            if ious:
                mean_iou = float(np.mean(ious))
                all_ious.append(mean_iou)
                results[key] = {'iou': mean_iou, 'n_frames': len(ious), 
                               'cam': CAM_ID, 'range': [start, end]}
                tqdm.write(f"  {subj}/{seq}: IoU={mean_iou:.3f} ({len(ious)} frames)")
            else:
                tqdm.write(f"  {subj}/{seq}: no contact frames")
                
        except Exception as e:
            tqdm.write(f"  {subj}/{seq}: ERROR {e}")
        
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    if all_ious:
        print(f"📊 HaPTIC on ARCTIC (third-person, cam {CAM_ID})")
        print(f"  Sequences: {len(all_ious)}")
        print(f"  Mean IoU: {np.mean(all_ious):.3f}")
        print(f"  Median IoU: {np.median(all_ious):.3f}")
        print(f"\n  Comparison:")
        print(f"    HaPTIC (3rd, OakInk):  0.340")
        print(f"    HaWoR  (1st, ARCTIC):  0.131")
        print(f"    HaPTIC (3rd, ARCTIC):  {np.mean(all_ious):.3f}")
    
    with open(RESULT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 {RESULT_JSON}")


if __name__ == "__main__":
    main()

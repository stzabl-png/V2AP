#!/usr/bin/env python3
"""
DP3 Inference HTTP Server — Baseline1
======================================
Cross-env bridge: DP3 runs in `dp3` conda env (torch 2.11.0+cu128, sm_120),
IsaacSim runs in `env_isaaclab`. They can't share Python — so this server
exposes the trained policy over HTTP for the sim process to call.

Usage:
    # In `dp3` env
    /home/accelerator/miniforge3/envs/dp3/bin/python Baseline1/eval/dp3_inference_server.py \\
        --ckpt /home/accelerator/UCB_Project/Baseline1/dp3_runs/all_combined_v2/checkpoints/epoch=0008-test_mean_score=-0.007.ckpt

    # Self-test only (no server, just sanity check against val episode)
    /home/accelerator/miniforge3/envs/dp3/bin/python Baseline1/eval/dp3_inference_server.py \\
        --ckpt <ckpt> --self-test-only

Client (in any env):
    import requests, numpy as np
    pc = np.random.randn(2, 4096, 3).astype(np.float32)   # (n_obs_steps, N, 3)
    ap = np.random.randn(2, 8).astype(np.float32)         # (n_obs_steps, 8)
    r = requests.post('http://127.0.0.1:8765/predict', json={
        'point_cloud': pc.tolist(),
        'agent_pos':   ap.tolist(),
    }).json()
    action      = np.array(r['action'])       # (n_action_steps, 8) = (8, 8) for default
    action_pred = np.array(r['action_pred'])  # (horizon, 8) = (16, 8) for default
"""
import os, sys, argparse, time, pathlib
import numpy as np

PROJ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "third_party" / "3D-Diffusion-Policy" / "3D-Diffusion-Policy"))

import torch
import dill
from omegaconf import OmegaConf

# import this to register the diffusion_policy_3d package + register the OmegaConf 'eval' resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.common.pytorch_util import dict_apply


def load_policy(ckpt_path: str, device: str = "cuda:0", use_ema: bool = True) -> DP3:
    """Load DP3 policy from training checkpoint."""
    print(f"loading ckpt {ckpt_path} ...")
    t0 = time.time()
    payload = torch.load(ckpt_path, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    print(f"  cfg.policy._target_ = {cfg.policy._target_}")
    print(f"  horizon={cfg.horizon} n_obs={cfg.n_obs_steps} n_action={cfg.n_action_steps}")

    # build model from cfg (this is what train.py does in __init__)
    import hydra
    model: DP3 = hydra.utils.instantiate(cfg.policy)

    # which state dict to load
    key = "ema_model" if (use_ema and "ema_model" in payload["state_dicts"]) else "model"
    print(f"  loading state_dict[{key}] ({len(payload['state_dicts'][key])} keys)")
    model.load_state_dict(payload["state_dicts"][key])
    model.to(device).eval()
    print(f"  ready in {time.time()-t0:.1f}s on {device}")
    return model, cfg


@torch.no_grad()
def predict(model: DP3, pc: np.ndarray, ap: np.ndarray, device: str = "cuda:0"):
    """
    Inputs (numpy, float32):
        pc: (n_obs_steps, N_points, 3)
        ap: (n_obs_steps, agent_dim)
    Returns (numpy, float32):
        action:      (n_action_steps, action_dim)
        action_pred: (horizon, action_dim)
    """
    obs_dict = {
        "point_cloud": torch.from_numpy(pc).unsqueeze(0).to(device),  # (1, T, N, 3)
        "agent_pos":   torch.from_numpy(ap).unsqueeze(0).to(device),  # (1, T, D)
    }
    result = model.predict_action(obs_dict)
    return (result["action"][0].cpu().numpy(),
            result["action_pred"][0].cpu().numpy())


# ── Sanity check against val zarr ────────────────────────────────────────────
def self_test(model: DP3, cfg, zarr_path: str = None):
    """Run policy on a val episode, compare predicted action to GT."""
    if zarr_path is None:
        zarr_path = "/home/accelerator/UCB_Project/Baseline1/data/human_dp_baseline_v2_all.zarr"
    print(f"\nself-test: reading val sample from {zarr_path}")
    import zarr
    r = zarr.open(zarr_path, "r")
    pc_arr    = r["data/point_cloud"]
    state_arr = r["data/state"]
    action_arr = r["data/action"]
    ep_ends   = r["meta/episode_ends"][:]

    n_obs = int(cfg.n_obs_steps); horizon = int(cfg.horizon)
    # pick val episode (last few are val per val_ratio=0.02 default split — but no strict guarantee;
    # for sanity any episode works since we just compare predicted vs GT, not "generalization")
    ep_idx = len(ep_ends) // 2
    ep_start = 0 if ep_idx == 0 else int(ep_ends[ep_idx - 1])
    ep_end   = int(ep_ends[ep_idx])
    print(f"  episode {ep_idx}: steps [{ep_start}, {ep_end})  T={ep_end-ep_start}")

    # take a window in the middle of the episode
    t0 = ep_start + (ep_end - ep_start) // 3
    pc = pc_arr[t0:t0 + n_obs]              # (n_obs, 4096, 3)
    ap = state_arr[t0:t0 + n_obs]           # (n_obs, 8)
    gt_action = action_arr[t0:t0 + horizon] # (horizon, 8)  ground truth

    print(f"  inference @ step {t0} ...")
    t_inf = time.time()
    action, action_pred = predict(model, pc.astype(np.float32), ap.astype(np.float32))
    print(f"    forward {time.time()-t_inf:.2f}s")
    print(f"    pred.action      shape={action.shape}")
    print(f"    pred.action_pred shape={action_pred.shape}")

    # quick MSE vs GT (action covers steps [n_obs-1 : n_obs-1+n_action])
    n_action = action.shape[0]
    start = n_obs - 1
    gt_action_aligned = gt_action[start:start + n_action]
    mse_xyz  = float(((action[:, :3]  - gt_action_aligned[:, :3])  ** 2).mean())
    mse_quat = float(((action[:, 3:7] - gt_action_aligned[:, 3:7]) ** 2).mean())
    mse_grip = float(((action[:, 7]   - gt_action_aligned[:, 7])   ** 2).mean())
    print(f"  vs GT (single-shot, model never saw this episode at this t):")
    print(f"    MSE xyz   = {mse_xyz:.6g}  ({np.sqrt(mse_xyz)*1000:.2f} mm RMSE)")
    print(f"    MSE quat  = {mse_quat:.6g}")
    print(f"    MSE grip  = {mse_grip:.6g}")

    # quick warm-loop timing
    print(f"  warm-loop timing: 5 forwards back-to-back ...")
    times = []
    for _ in range(5):
        t = time.time()
        predict(model, pc.astype(np.float32), ap.astype(np.float32))
        times.append(time.time() - t)
    print(f"    avg {np.mean(times)*1000:.1f} ms  min {np.min(times)*1000:.1f} ms")


# ── HTTP server (FastAPI) ────────────────────────────────────────────────────
def build_app(model: DP3, cfg):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List

    app = FastAPI(title="DP3 Inference (Baseline1)")

    class PredictReq(BaseModel):
        point_cloud: List         # (n_obs_steps, N, 3)
        agent_pos:   List         # (n_obs_steps, 8)

    class PredictResp(BaseModel):
        action:      List         # (n_action_steps, 8)
        action_pred: List         # (horizon, 8)
        inference_ms: float

    @app.get("/info")
    def info():
        return {
            "horizon":        int(cfg.horizon),
            "n_obs_steps":    int(cfg.n_obs_steps),
            "n_action_steps": int(cfg.n_action_steps),
            "action_dim":     int(cfg.policy.shape_meta.action.shape[0]),
            "point_cloud_shape": list(cfg.policy.shape_meta.obs.point_cloud.shape),
            "agent_pos_dim":  int(cfg.policy.shape_meta.obs.agent_pos.shape[0]),
        }

    @app.post("/predict", response_model=PredictResp)
    def post_predict(req: PredictReq):
        try:
            pc = np.asarray(req.point_cloud, dtype=np.float32)
            ap = np.asarray(req.agent_pos,   dtype=np.float32)
        except Exception as e:
            raise HTTPException(400, f"bad input dtype/shape: {e}")
        # basic shape check
        if pc.ndim != 3 or pc.shape[2] != 3:
            raise HTTPException(400, f"point_cloud must be (T,N,3), got {pc.shape}")
        if ap.ndim != 2:
            raise HTTPException(400, f"agent_pos must be (T,D), got {ap.shape}")
        if pc.shape[0] != ap.shape[0]:
            raise HTTPException(400, f"T mismatch: pc {pc.shape[0]} vs ap {ap.shape[0]}")

        t0 = time.time()
        action, action_pred = predict(model, pc, ap)
        return PredictResp(action=action.tolist(),
                           action_pred=action_pred.tolist(),
                           inference_ms=(time.time() - t0) * 1000.0)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-ema", action="store_true", help="use raw model instead of ema_model")
    ap.add_argument("--self-test-only", action="store_true", help="run sanity check and exit (no server)")
    args = ap.parse_args()

    model, cfg = load_policy(args.ckpt, device=args.device, use_ema=not args.no_ema)
    self_test(model, cfg)

    if args.self_test_only:
        print("\nself-test done; --self-test-only ⇒ exit without starting server.")
        return

    print(f"\nstarting HTTP server on http://{args.host}:{args.port}")
    print(f"  POST /predict   with body {{point_cloud: [T,N,3], agent_pos: [T,D]}}")
    print(f"  GET  /info      shape metadata")
    app = build_app(model, cfg)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

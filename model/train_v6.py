#!/usr/bin/env python3
"""
PointNet++ v6 Training — Continuous Affordance Heatmap

Input: xyz+normals (6ch), 4096 points
Output: per-point soft heatmap [0,1] via sigmoid
Loss: Weighted L1
Metrics: Pearson Correlation, MAE, Peak IoU

Usage:
    python -m model.train_v6 --epochs 300
"""

import os, sys, json, argparse, time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast, GradScaler  # PyTorch >= 2.4
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch 2.1–2.3
    _AMP_DEVICE = None


def _autocast_cuda():
    if _AMP_DEVICE is not None:
        return autocast(_AMP_DEVICE, dtype=torch.float16)
    return autocast(dtype=torch.float16)


def _grad_scaler():
    if _AMP_DEVICE is not None:
        return GradScaler(_AMP_DEVICE)
    return GradScaler()


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pointnet2_v6 import PointNet2AffordanceV6
from model.losses_v6 import WeightedL1HeatmapLoss
from model.metrics_v6 import compute_all_metrics, threshold_search_v6


# ============================================================
# Dataset
# ============================================================

class AffordanceHeatmapDataset(Dataset):
    """HDF5 dataset for continuous affordance heatmap regression.

    Reads from affordance_*_soft.h5 files with soft_labels field.
    Augmentation A: Point Dropout (30% prob, drop 10%)
    Augmentation B: Multi-View Subsample (random 2048~4096 points)
    """

    def __init__(self, h5_path, obj_ids_to_use=None, augment=True, num_points=4096,
                 oversample_factor=1):
        self.augment = augment
        self.num_points = num_points
        self.oversample_factor = oversample_factor

        with h5py.File(h5_path, 'r') as f:
            all_points = f['data/points'][:]
            all_normals = f['data/normals'][:]
            all_obj_ids = f['data/obj_ids'][:]

            # Use soft_labels if available, else fall back to binary labels
            if 'data/soft_labels' in f:
                all_labels = f['data/soft_labels'][:]
            else:
                all_labels = f['data/labels'][:]

        if obj_ids_to_use is not None:
            decoded = [s.decode() if isinstance(s, bytes) else s for s in all_obj_ids]
            mask = np.array([oid in obj_ids_to_use for oid in decoded])
            self.points = all_points[mask]
            self.normals = all_normals[mask]
            self.labels = all_labels[mask]
        else:
            self.points = all_points
            self.normals = all_normals
            self.labels = all_labels

        self.num_samples = len(self.points)
        n_objs = len(obj_ids_to_use) if obj_ids_to_use else "all"
        avg_label = self.labels.mean() if self.num_samples > 0 else 0.0
        print(f"    Loaded {self.num_samples} samples ({n_objs} objects), "
              f"avg_soft_label={avg_label:.4f}")

    def __len__(self):
        return self.num_samples * self.oversample_factor

    def __getitem__(self, idx):
        real_idx = idx % self.num_samples
        pts = self.points[real_idx].copy()    # (N, 3)
        nrm = self.normals[real_idx].copy()   # (N, 3)
        lbl = self.labels[real_idx].copy()    # (N,)

        if self.augment:
            pts, nrm, lbl = self._augment(pts, nrm, lbl)

        features = pts.copy()  # (N, 3) — xyz only
        return (
            torch.from_numpy(pts).float(),
            torch.from_numpy(features).float(),
            torch.from_numpy(lbl).float(),
        )

    def _augment(self, pts, nrm, lbl):
        N = len(pts)

        # === Aug B: Multi-View Subsample (50% prob) ===
        if np.random.rand() < 0.5:
            n_keep = np.random.randint(N // 2, N)  # 2048~4096
            keep_idx = np.random.choice(N, n_keep, replace=False)
            pts_sub = pts[keep_idx]
            nrm_sub = nrm[keep_idx]
            lbl_sub = lbl[keep_idx]
            # Pad back to N by duplicating random points
            if n_keep < N:
                pad_idx = np.random.choice(n_keep, N - n_keep, replace=True)
                pts = np.concatenate([pts_sub, pts_sub[pad_idx]], axis=0)
                nrm = np.concatenate([nrm_sub, nrm_sub[pad_idx]], axis=0)
                lbl = np.concatenate([lbl_sub, lbl_sub[pad_idx]], axis=0)
            else:
                pts, nrm, lbl = pts_sub, nrm_sub, lbl_sub

        # === SO(3) random rotation ===
        z = np.random.randn(3, 3).astype(np.float32)
        q, r = np.linalg.qr(z)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        R = (q @ np.diag(ph)).astype(np.float32)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        pts = pts @ R.T
        nrm = nrm @ R.T

        # === Random scale [0.8, 1.2] ===
        scale = np.random.uniform(0.8, 1.2)
        pts *= scale

        # === Random shift [-0.02, 0.02] ===
        shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
        pts += shift

        # === Jitter σ=0.002 ===
        pts += np.random.normal(0, 0.002, size=pts.shape).astype(np.float32)

        # === Aug A: Point Dropout (30% prob, drop 10%) ===
        if np.random.rand() < 0.3:
            n = len(pts)
            keep = np.random.choice(n, int(n * 0.9), replace=False)
            drop = np.setdiff1d(np.arange(n), keep)
            fill = np.random.choice(keep, len(drop), replace=True)
            pts[drop] = pts[fill]
            nrm[drop] = nrm[fill]
            lbl[drop] = lbl[fill]

        return pts, nrm, lbl


# ============================================================
# Training / Eval Loops
# ============================================================

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_metrics = {"pearson": 0, "mae": 0, "peak_iou_03": 0, "peak_iou_05": 0}
    n = 0

    for xyz, features, target in loader:
        xyz = xyz.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_cuda():
            pred = model(xyz, features)
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            m = compute_all_metrics(pred.float(), target)
        for k in all_metrics:
            all_metrics[k] += m[k]
        n += 1

    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"pearson": 0, "mae": 0, "peak_iou_03": 0, "peak_iou_05": 0}
    n = 0

    for xyz, features, target in loader:
        xyz = xyz.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with _autocast_cuda():
            pred = model(xyz, features)
            loss = criterion(pred, target)

        total_loss += loss.item()
        m = compute_all_metrics(pred.float(), target)
        for k in all_metrics:
            all_metrics[k] += m[k]
        n += 1

    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


# ============================================================
# Visualization
# ============================================================

def save_vis(model, dataset, device, path, epoch, history=None):
    model.eval()
    n_samples = min(4, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    n_rows = n_samples + (1 if history else 0)
    fig = plt.figure(figsize=(18, 5 * n_rows))

    for row, idx in enumerate(indices):
        pts_t, feat_t, lbl_t = dataset[idx]
        pts = pts_t.numpy()
        gt = lbl_t.numpy()

        with torch.no_grad(), _autocast_cuda():
            pred = model(
                pts_t.unsqueeze(0).to(device),
                feat_t.unsqueeze(0).to(device),
            ).float().cpu().squeeze(0).numpy()

        cmap = plt.cm.jet
        for col, (title, vals) in enumerate([
            (f'GT heatmap (max={gt.max():.2f})', gt),
            (f'Pred heatmap (max={pred.max():.2f})', pred),
            (f'|Error| (MAE={np.abs(pred-gt).mean():.4f})', np.abs(pred - gt)),
        ]):
            ax = fig.add_subplot(n_rows, 3, row * 3 + col + 1, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=cmap(vals)[:, :3], s=2, alpha=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    if history and len(history) > 1:
        epochs = [h['epoch'] for h in history]

        ax1 = fig.add_subplot(n_rows, 3, n_samples * 3 + 1)
        ax1.plot(epochs, [h['train_loss'] for h in history], label='Train', c='blue')
        ax1.plot(epochs, [h['val_loss'] for h in history], label='Val', c='red')
        ax1.set_title('Loss (Weighted L1)'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(n_rows, 3, n_samples * 3 + 2)
        ax2.plot(epochs, [h['val_pearson'] for h in history], c='green', lw=2)
        best_p = max(h['val_pearson'] for h in history)
        ax2.set_title(f'Val Pearson (best={best_p:.4f})'); ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(n_rows, 3, n_samples * 3 + 3)
        ax3.plot(epochs, [h['val_mae'] for h in history], c='orange', lw=2)
        ax3.set_title(f'Val MAE'); ax3.grid(True, alpha=0.3)

    fig.suptitle(f'v6 Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"        📊 Vis saved: {os.path.basename(path)}")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description="PointNet++ v6 Affordance Heatmap Training")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--fg_weight", type=float, default=5.0,
                    help="Loss weight for foreground (soft_label >= 0.05) regions")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (experimental; PyTorch 2.1 + FPS often fails)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 avoids CUDA+fork deadlock; 4+ faster IO)",
    )
    args = p.parse_args()

    # Paths
    if args.dataset_dir is None:
        args.dataset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_hub", "training_m5"
        )
    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "checkpoints_v6"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PointNet++ v6 — Continuous Affordance Heatmap")
    print("=" * 70)
    print(f"  Input:       xyz only (3ch), 4096 pts")
    print(f"  Output:      sigmoid heatmap [0,1]")
    print(f"  Loss:        Weighted L1 (fg_weight={args.fg_weight})")
    print(f"  Device:      {device}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr} + {args.warmup_epochs}ep warmup")
    print(f"  AMP:         ✅ float16")
    print(f"  Compile:     {'✅ torch.compile' if args.compile else '❌ off (default)'}")
    print(f"  DataLoader:  num_workers={args.num_workers}")
    print(f"  Dataset:     {args.dataset_dir}")
    print(f"  Checkpoints: {args.save_dir}")
    sys.stdout.flush()

    # ============================================================
    # Data split (use pre-computed split from HuggingFace)
    # ============================================================
    split_path = os.path.join(args.dataset_dir, "objects_train_val_split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        train_obj_ids = set(split["train"])
        val_obj_ids = set(split["val"])
        print(f"\n  Using pre-computed split: {len(train_obj_ids)} train / {len(val_obj_ids)} val")
    else:
        # Fallback: auto split
        print(f"\n  ⚠️ No split file found, using auto 80/20 split")
        train_h5 = os.path.join(args.dataset_dir, "affordance_train_soft.h5")
        with h5py.File(train_h5, 'r') as f:
            ids = sorted(set(
                s.decode() if isinstance(s, bytes) else s
                for s in f['data/obj_ids'][:]
            ))
        np.random.seed(42)
        np.random.shuffle(ids)
        n_val = max(1, int(len(ids) * 0.2))
        val_obj_ids = set(ids[:n_val])
        train_obj_ids = set(ids[n_val:])

    overlap = train_obj_ids & val_obj_ids
    assert len(overlap) == 0, f"Train/Val overlap: {overlap}"
    print(f"  ✅ Zero overlap")
    sys.stdout.flush()

    # ============================================================
    # Load data (soft label versions)
    # ============================================================
    train_soft_h5 = os.path.join(args.dataset_dir, "affordance_train_soft.h5")
    val_soft_h5 = os.path.join(args.dataset_dir, "affordance_val_soft.h5")

    # Fallback to non-soft if soft files don't exist
    if not os.path.exists(train_soft_h5):
        train_soft_h5 = os.path.join(args.dataset_dir, "affordance_train.h5")
        val_soft_h5 = os.path.join(args.dataset_dir, "affordance_val.h5")
        print(f"  ⚠️ Soft label files not found, using binary labels")

    print(f"\n  Loading training data...")
    train_ds = AffordanceHeatmapDataset(train_soft_h5, train_obj_ids, augment=True,
                                         oversample_factor=20)
    print(f"  Loading val data...")
    val_ds = AffordanceHeatmapDataset(val_soft_h5, val_obj_ids, augment=False)

    # Cross-load: val objects from train file, train objects from val file
    print(f"  Cross-loading val objects from train file...")
    val_from_train = AffordanceHeatmapDataset(train_soft_h5, val_obj_ids, augment=False)
    if val_from_train.num_samples > 0:
        val_ds.points = np.concatenate([val_ds.points, val_from_train.points])
        val_ds.normals = np.concatenate([val_ds.normals, val_from_train.normals])
        val_ds.labels = np.concatenate([val_ds.labels, val_from_train.labels])
        val_ds.num_samples = len(val_ds.points)

    print(f"  Cross-loading train objects from val file...")
    train_from_val = AffordanceHeatmapDataset(val_soft_h5, train_obj_ids, augment=True)
    if train_from_val.num_samples > 0:
        train_ds.points = np.concatenate([train_ds.points, train_from_val.points])
        train_ds.normals = np.concatenate([train_ds.normals, train_from_val.normals])
        train_ds.labels = np.concatenate([train_ds.labels, train_from_val.labels])
        train_ds.num_samples = len(train_ds.points)

    print(f"\n  Summary:")
    print(f"    Train: {len(train_ds)} samples, {len(train_obj_ids)} objects")
    print(f"    Val:   {len(val_ds)} samples, {len(val_obj_ids)} objects")
    sys.stdout.flush()

    loader_kw = dict(
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        loader_kw["num_workers"] = args.num_workers
        loader_kw["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, **loader_kw,
    )

    # ============================================================
    # Model
    # ============================================================
    model = PointNet2AffordanceV6(in_channel=3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params: {n_params:,}")

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print(f"  ✅ torch.compile enabled")
        except Exception as e:
            print(f"  ⚠️ torch.compile failed: {e}")
    elif args.compile:
        print(f"  ⚠️ torch.compile not available in this PyTorch build")

    criterion = WeightedL1HeatmapLoss(bg_weight=1.0, fg_weight=args.fg_weight, threshold=0.05)
    print(f"  Loss: WeightedL1 (bg=1.0, fg={args.fg_weight}, thresh=0.05)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Warmup + Cosine schedule
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    scaler = _grad_scaler()

    print(f"\n{'='*90}")
    print(f"{'Ep':>4} | {'Loss':>8} | {'V.Loss':>8} | "
          f"{'Pearson':>8} | {'MAE':>8} | {'IoU@.3':>7} | {'IoU@.5':>7} | {'LR':>9}")
    print(f"{'-'*90}")
    sys.stdout.flush()

    best_pearson = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        t_loss, t_met = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        v_loss, v_met = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"{epoch:>4} | {t_loss:>8.5f} | {v_loss:>8.5f} | "
              f"{v_met['pearson']:>8.4f} | {v_met['mae']:>8.5f} | "
              f"{v_met['peak_iou_03']:>6.1%} | {v_met['peak_iou_05']:>6.1%} | "
              f"{lr:>9.2e}  ({elapsed:.0f}s)")
        sys.stdout.flush()

        history.append({
            "epoch": epoch,
            "train_loss": round(t_loss, 6),
            "val_loss": round(v_loss, 6),
            "val_pearson": round(v_met['pearson'], 5),
            "val_mae": round(v_met['mae'], 6),
            "val_peak_iou_03": round(v_met['peak_iou_03'], 4),
            "val_peak_iou_05": round(v_met['peak_iou_05'], 4),
            "lr": round(lr, 8),
            "time_s": round(elapsed, 1),
        })

        # Save best by Pearson correlation
        if v_met['pearson'] > best_pearson:
            best_pearson = v_met['pearson']
            ckpt = {
                'epoch': epoch,
                'model_state_dict': (model._orig_mod.state_dict()
                                     if hasattr(model, '_orig_mod') else
                                     model.state_dict()),
                'val_pearson': best_pearson,
                'val_mae': v_met['mae'],
                'val_peak_iou_03': v_met['peak_iou_03'],
                'train_objects': sorted(train_obj_ids),
                'val_objects': sorted(val_obj_ids),
                'version': 'v6',
            }
            torch.save(ckpt, os.path.join(args.save_dir, "best_v6_model.pth"))
            print(f"        ★ New best! Pearson={best_pearson:.4f} MAE={v_met['mae']:.5f}")
            sys.stdout.flush()

        # Periodic checkpoint + visualization
        if epoch % 20 == 0 or epoch == 1:
            state = (model._orig_mod.state_dict()
                     if hasattr(model, '_orig_mod') else model.state_dict())
            torch.save({'epoch': epoch, 'model_state_dict': state},
                       os.path.join(args.save_dir, f"checkpoint_ep{epoch}.pth"))

            # Visualization (use uncompiled model if compiled)
            vis_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            save_vis(vis_model, val_ds, device,
                     os.path.join(args.save_dir, f"vis_ep{epoch}.png"),
                     epoch, history)

    # ============================================================
    # Final
    # ============================================================
    state = (model._orig_mod.state_dict()
             if hasattr(model, '_orig_mod') else model.state_dict())
    torch.save({'epoch': args.epochs, 'model_state_dict': state, 'version': 'v6'},
               os.path.join(args.save_dir, "final_v6_model.pth"))

    with open(os.path.join(args.save_dir, "training_history_v6.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # Threshold search
    print(f"\n--- Post-training threshold search ---")
    vis_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    best_thresh, best_iou, final_pearson = threshold_search_v6(vis_model, val_loader, device)

    info = {
        "version": "v6",
        "train_objects": sorted(train_obj_ids),
        "val_objects": sorted(val_obj_ids),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_pearson": round(best_pearson, 5),
        "best_threshold": round(best_thresh, 3),
        "best_peak_iou": round(best_iou, 4),
        "num_points": 4096,
        "in_channel": 6,
        "fg_weight": args.fg_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    with open(os.path.join(args.save_dir, "run_info_v6.json"), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE (v6)")
    print(f"  Best Pearson: {best_pearson:.4f}")
    print(f"  Best thresh:  {best_thresh:.2f} → IoU={best_iou:.1%}")
    print(f"  Model:        {args.save_dir}/best_v6_model.pth")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

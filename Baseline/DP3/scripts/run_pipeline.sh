#!/bin/bash
# ============================================================
# Baseline2 — 一键运行完整 Pipeline (OakInk)
# ============================================================
set -e

PROJ="/home/lyh/Project/V2AP"
DATASET=${1:-oakink}

echo "============================================================"
echo "  Baseline2 完整 Pipeline"
echo "  Dataset: $DATASET"
echo "============================================================"

# Step 1: Sim Replay 记录轨迹
echo ""
echo "[Step 1/3] Sim Trajectory Recording..."
sim45 "$PROJ/sim/record_trajectory.py" \
    --all \
    --dataset "$DATASET" \
    --headless

# Step 2: HDF5 → Zarr
echo ""
echo "[Step 2/3] HDF5 → Zarr..."
python "$PROJ/Baseline2/data_pipeline/hdf5_to_zarr.py" \
    --dataset "$DATASET"

# Step 3: DP3 训练
echo ""
echo "[Step 3/3] DP3 Training..."
bash "$PROJ/Baseline2/scripts/train_dp3.sh" 0 0 "$DATASET"

echo ""
echo "============================================================"
echo "  ✅ Baseline2 Pipeline 完成!"
echo "============================================================"

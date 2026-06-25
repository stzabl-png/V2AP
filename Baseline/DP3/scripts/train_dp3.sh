#!/bin/bash
# ============================================================
# Baseline2 — Step 3: DP3 训练
# ============================================================
set -e

SEED=${1:-0}
GPU=${2:-0}
DATASET=${3:-oakink}

PROJ="/home/lyh/Project/V2AP"
DP3_DIR="/home/lyh/Project/Sim_VideoPolicy/dp3"
ZARR_PATH="$PROJ/Baseline2/data/zarr/$DATASET"
TASK="sim_grasping"
DATE=$(date +%y%m%d)

echo "============================================================"
echo "  Baseline2 — DP3 Training"
echo "  Dataset:  $DATASET"
echo "  Zarr:     $ZARR_PATH"
echo "  Seed:     $SEED   GPU: $GPU"
echo "  Date:     $DATE"
echo "============================================================"

if [ ! -d "$ZARR_PATH" ]; then
    echo "❌ Zarr 数据不存在: $ZARR_PATH"
    echo "   请先运行:"
    echo "   sim45 sim/record_trajectory.py --all --dataset $DATASET --headless"
    echo "   python Baseline2/data_pipeline/hdf5_to_zarr.py --dataset $DATASET"
    exit 1
fi

cd "$DP3_DIR" || exit 1
export CUDA_VISIBLE_DEVICES=${GPU}

python train.py \
    --config-name=train_diffusion_unet_hybrid \
    task=${TASK} \
    task.dataset.zarr_path=${ZARR_PATH} \
    training.seed=${SEED} \
    training.device="cuda:0" \
    exp_name="baseline2-${DATASET}-${DATE}" \
    logging.mode=online \
    checkpoint.save_ckpt=true \
    checkpoint.save_last_ckpt=true

echo ""
echo "✅ 训练完成!"
echo "   Checkpoint: $DP3_DIR/output/baseline2-${DATASET}-${DATE}/"

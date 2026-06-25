#!/bin/bash
# =============================================================================
# full_auto_collect.sh — 全自动抓取数据收集主循环
# =============================================================================
# 流程：
#   Step A: 生成候选池 (HP约束, 50个/物体)
#   Step B: Sim验证 (8 yaw × 10 xy位置, 两张GPU各4进程)
#   Step C: 合并结果 → 自动回到 Step A
#
# 用法:
#   export ISAAC_SIM_PATH=/path/to/isaac-sim
#   bash scripts/full_auto_collect.sh
#   bash scripts/full_auto_collect.sh --outdir output/collect_v2 --max-rounds 20
# =============================================================================

set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN=/home/vision/miniconda3/envs/bundlesdf/bin/python3

# ── 参数默认值 ────────────────────────────────────────────────────────────────
OUTDIR="${OUTDIR:-$PROJ/output/grasp_collect_v2}"
MAX_ROUNDS="${MAX_ROUNDS:-999}"        # 无限直到手动 Ctrl+C
TARGET_CANDIDATES=50                   # 每物体候选池大小
GPU_IDS="0,1"                          # 两张 TitanX
SIM_PER_GPU=4                          # 每GPU 4进程
SLOTS_PER_ROUND="${SLOTS_PER_ROUND:-500}"
HEADLESS="--headless"
DATASET="${DATASET:-}"                 # 留空 = 所有数据集

: "${ISAAC_SIM_PATH:?必须设置 ISAAC_SIM_PATH 环境变量}"

mkdir -p "$OUTDIR"

echo "============================================================"
echo "  V2AP 全自动抓取数据收集"
echo "  输出目录: $OUTDIR"
echo "  GPUs: $GPU_IDS  (${SIM_PER_GPU}进程/GPU)"
echo "  Yaw:  0/45/90/135/180/225/270/315° (8个)"
echo "  XY:   ±10cm (10次随机位置/物体/yaw)"
echo "  候选池: $TARGET_CANDIDATES 个/物体"
echo "  最大轮数: $MAX_ROUNDS"
echo "============================================================"

ROUND=0

while [ $ROUND -lt $MAX_ROUNDS ]; do
    ROUND=$((ROUND + 1))
    echo ""
    echo "================================================================"
    echo "  轮次 $ROUND / $MAX_ROUNDS — $(date)"
    echo "================================================================"

    # ── Step A: 生成候选池 ───────────────────────────────────────────
    echo ""
    echo "  [Step A] 生成 HP 约束候选池 (target=$TARGET_CANDIDATES)..."
    DS_ARGS=""
    if [ -n "$DATASET" ]; then
        DS_ARGS="--dataset $DATASET"
    fi

    $PYBIN scripts/batch_gen_candidates_pool.py \
        --outdir "$OUTDIR" \
        --target "$TARGET_CANDIDATES" \
        --success-threshold 5 \
        --gen-mode mixed \
        --min-merged-success 1 \
        --resume \
        $DS_ARGS \
        || { echo "  ⚠️  候选生成警告，继续..."; }

    # ── Step B: Sim 验证 (8 yaw × 10 xy) ──────────────────────────
    echo ""
    echo "  [Step B] Sim 验证 (8 yaw × 10 pos)..."

    $PYBIN scripts/batch_sim_candidates_pool.py \
        --outdir "$OUTDIR" \
        --sim-gpu-ids "$GPU_IDS" \
        --sim-per-gpu "$SIM_PER_GPU" \
        --slots-per-round "$SLOTS_PER_ROUND" \
        --max-rounds 1 \
        --yaw-grid "0,45,90,135,180,225,270,315" \
        --xy-positions 10 \
        --xy-jitter-m 0.10 \
        --resume \
        $HEADLESS \
        || { echo "  ⚠️  Sim 轮次完成（可能有部分失败）"; }

    echo "  [Step B] 完成 — $(date)"
    echo ""
    echo "  等待 5 秒后开始下一轮..."
    sleep 5
done

echo ""
echo "============================================================"
echo "  全自动收集完成: $ROUND 轮"
echo "  结果: $OUTDIR"
echo "============================================================"

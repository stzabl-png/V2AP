#!/bin/bash
# ============================================================
# run_sim_local_dexycb.sh  —  本机 RTX 5090: DexYCB Sim 抓取
# ============================================================
set -e

PROJ="/home/lyh/Project/V2AP"
ISAAC_SIM_PATH="/home/lyh/isaac-sim"   # sim45 (4.5.0-rc.36, 已验证)
SIM_SCRIPT="$PROJ/sim/run_grasp_sim.py"

GRASP_DIR="${GRASP_DIR:-$PROJ/output/grasps_candidate_dexycb}"
GT_DIR="${GT_DIR:-$PROJ/output/robot_gt_dexycb}"
LOG_DIR="${LOG_DIR:-$PROJ/output/sim_logs_dexycb}"

export PYTHONUNBUFFERED=1

mkdir -p "$GT_DIR" "$LOG_DIR"

HDF5_LIST=($(ls "$GRASP_DIR"/*_grasp.hdf5 2>/dev/null | sort))
TOTAL=${#HDF5_LIST[@]}

echo "============================================================"
echo "  本机 RTX 5090 — DexYCB Sim 抓取验证"
echo "  Isaac Sim:  $ISAAC_SIM_PATH"
echo "  候选目录:   $GRASP_DIR"
echo "  结果目录:   $GT_DIR"
echo "  物体总数:   $TOTAL"
echo "  开始时间:   $(date)"
echo "============================================================"

SUCCESS=0; FAILED=0; SKIPPED=0

for HDF5 in "${HDF5_LIST[@]}"; do
    OBJ_ID=$(basename "$HDF5" _grasp.hdf5)
    RESULT_FILE="$GT_DIR/${OBJ_ID}_robot_gt.hdf5"
    N=$((SUCCESS + FAILED + SKIPPED + 1))

    if [ -f "$RESULT_FILE" ]; then
        echo "  [$N/$TOTAL] $OBJ_ID → ⏭️ 已完成，跳过"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo ""
    echo "  [$N/$TOTAL] $OBJ_ID ..."

    timeout 600 "$ISAAC_SIM_PATH/python.sh" "$SIM_SCRIPT" \
        --hdf5 "$HDF5" \
        --headless \
        --save-result \
        --result-dir "$GT_DIR" \
        2>&1 | tee "$LOG_DIR/${OBJ_ID}.log" \
             | grep --line-buffered -E \
               "Attempt [0-9]+|Plan OK|Plan FAILED|GRASP SUCCESS|GRASP FAILED|✅|❌|score=|Saved:|ERROR" \
        || true

    if [ -f "$RESULT_FILE" ]; then
        N_SUC=$(python3 -c "
import h5py
with h5py.File('$RESULT_FILE','r') as f:
    print(int(f.attrs.get('n_successful', 0)))
" 2>/dev/null || echo "0")
        if [ "$N_SUC" -gt 0 ]; then
            echo "  ✅ $OBJ_ID → 成功 ($N_SUC 个候选)"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  ❌ $OBJ_ID → 失败"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ❌ $OBJ_ID → 超时或崩溃"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  DexYCB 完成: ✅$SUCCESS  ❌$FAILED  ⏭️$SKIPPED / $TOTAL"
echo "  结束时间: $(date)"
echo "  结果: $GT_DIR"
echo "============================================================"

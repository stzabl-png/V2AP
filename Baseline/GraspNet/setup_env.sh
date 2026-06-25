#!/usr/bin/env bash
# ============================================================
# GraspNet-Baseline — Environment Setup
# ============================================================
# Run from project root:  bash Baseline2/graspnet/setup_env.sh
#
# Prerequisites:
#   - conda installed
#   - CUDA toolkit accessible (nvcc in PATH)
# ============================================================
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
GRASPNET_DIR="$PROJ/third_party/graspnet-baseline"

echo "=== Step 1: Clone graspnet-baseline ==="
if [ ! -d "$GRASPNET_DIR" ]; then
    git clone https://github.com/graspnet/graspnet-baseline.git "$GRASPNET_DIR"
else
    echo "  ✅ Already cloned at $GRASPNET_DIR"
fi

echo ""
echo "=== Step 2: Create conda environment ==="
if conda env list | grep -q "^graspnet "; then
    echo "  ✅ conda env 'graspnet' already exists"
else
    conda create -n graspnet python=3.9 -y
fi

echo ""
echo "=== Step 3: Install dependencies ==="
echo "  Activate with: conda activate graspnet"
echo "  Then run:"
echo ""
cat << 'INSTALL_EOF'
    # PyTorch (adjust CUDA version as needed)
    pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121

    # GraspNet dependencies
    pip install graspnetAPI open3d trimesh scipy h5py termcolor Pillow

    # Compile PointNet++ CUDA ops
    cd third_party/graspnet-baseline/pointnet2
    python setup.py install
    cd ..

    # Compile KNN CUDA ops
    cd knn
    python setup.py install
    cd ../..

    # If PointNet++ compile fails, try community package:
    # pip install pointnet2_ops
INSTALL_EOF

echo ""
echo "=== Step 4: Download pretrained weights ==="
CKPT_DIR="$PROJ/Baseline2/graspnet/checkpoints"
if [ -f "$CKPT_DIR/checkpoint-rs.tar" ]; then
    echo "  ✅ checkpoint-rs.tar already exists"
else
    echo "  ⚠️  Download checkpoint-rs.tar from GraspNet Google Drive:"
    echo "     https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view"
    echo "     Save to: $CKPT_DIR/checkpoint-rs.tar"
fi

echo ""
echo "=== Setup complete ==="
echo "Next: conda activate graspnet && python Baseline2/graspnet/graspnet_infer.py --help"

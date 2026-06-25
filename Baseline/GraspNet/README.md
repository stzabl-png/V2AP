# Baseline2: GraspNet-Baseline

> **Paper**: *GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping* (CVPR 2020)
> **Method**: Pure geometry-based 6-DoF grasp detection — no affordance, no human prior.

## Purpose

This baseline demonstrates what a **state-of-the-art geometric grasp detector** achieves
on our evaluation objects, without any human-prior guidance. The performance gap between
GraspNet and our A2G method quantifies the value of affordance-driven grasping.

## Quick Start

```bash
# 1. Setup (one-time)
bash Baseline2/graspnet/setup_env.sh

# 2. Activate environment
conda activate graspnet

# 3. Batch inference on evaluation objects
python Baseline2/graspnet/batch_graspnet.py \
    --eval-csv evaluation/configs/eval_objects_merged_success_ge40.csv \
    --checkpoint Baseline2/graspnet/checkpoints/checkpoint-rs.tar

# 4. Run eval_pool (in IsaacSim environment)
sim45 evaluation/eval_pool.py \
    --policy graspnet_baseline \
    --candidate-dir output/graspnet_candidates \
    --eval-csv evaluation/configs/eval_objects_merged_success_ge40.csv \
    --result-dir output/eval_graspnet
```

## File Structure

```
Baseline2/graspnet/
├── README.md               # This file
├── setup_env.sh            # Environment setup script
├── graspnet_infer.py       # Single-object GraspNet inference
├── graspnet_to_hdf5.py     # Coordinate conversion + HDF5 writer
├── batch_graspnet.py       # Batch inference for all eval objects
└── checkpoints/            # Place checkpoint-rs.tar here
```

## Data Requirements

From HuggingFace `UCBProject/eval_assets`:
- `data_hub/meshes/SAM3DMesh/rotated_mesh/` — Object meshes
- `data_hub/ProcessedData/obj_meshes/*/scale.json` — Metric scale metadata

## Coordinate System Notes

GraspNet outputs grasps in metric (scaled) coordinates. Our pipeline converts them:
- **Rotation**: GraspNet `[-approach, finger, binormal]` → A2G `[finger, binormal, approach]`
- **Position**: GraspNet gripper-base center → A2G contact midpoint (`center + approach × depth`)
- **Scale**: Metric → mesh coordinates (÷ scale_factor)

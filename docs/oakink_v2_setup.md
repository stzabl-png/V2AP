# OakInk v2 Integration Guide

End-to-end workflow for OakInk-v2 → V2AP on a fresh machine.

## Prerequisites
- V2AP cloned and runnable
- Python 3.10+
- MANO models downloaded to `~/Project/mano_v1_2/`

---

## Step 1 · Install Dependencies

```bash
pip install oakink2-toolkit manotorch chumpy huggingface_hub
```

If Python >= 3.12, fix chumpy compatibility:
```bash
# Find chumpy install path
python -c "import chumpy; print(chumpy.__file__)"
# In ch.py, replace inspect.getargspec with inspect.getfullargspec
```

## Step 2 · Download OakInk-v2 Data (~37 GB)

```bash
mkdir -p ~/Project/OakInk2 && cd ~/Project/OakInk2
git clone https://github.com/oakink/OakInk2.git OakInk-v2-hub
cd OakInk-v2-hub

# Download annotations
python script/download.py
# Select:
#   - anno_preview  (annotation data, required)
#   - object_raw    (object meshes, required)
#   - object_affordance (optional)
```

## Step 3 · Extract Contact Maps (~21 hours)

```bash
cd ~/Project/V2AP
nohup python data/extract_contacts_v2.py \
  --oakink2_dir ~/Project/OakInk2/OakInk-v2-hub \
  --workers 4 \
  > extract_v2_full.log 2>&1 &

# Monitor progress
tail -f extract_v2_full.log
```

Output: ~57,000 NPZ files in `output/contacts_v2/`.
Supports resume — rerun the same command to continue from where it stopped.

## Step 4 · Build Merged Training Set (~30 min)

```bash
cd ~/Project/V2AP
python data/build_dataset.py --output_dir output/dataset_v1v2
```

Output: `output/dataset_v1v2/affordance_train.h5` + `affordance_val.h5`
Automatically merges `output/contacts/` (v1) + `output/contacts_v2/` (v2).

## Step 5 · Train Model (~2 hours)

```bash
python model/train_v5.py --save_dir output/checkpoints_v1v2
```

Output: `output/checkpoints_v1v2/best_model.pth` (F1 ~44%)

## Step 6 · Generate Grasp Poses with New Model

```bash
# Set as default model
cp output/checkpoints_v1v2/best_model.pth output/checkpoints/best_model.pth

# Batch generate grasps for v1 objects
python batch_process.py --force

# Batch generate grasps for v2 objects
python batch_process_v2.py --force
```

Output: `output/grasps/*.hdf5` (one file per object)

## Step 7 · USD Conversion + Sim Test

```bash
# OBJ → USD (requires Isaac Sim)
sim45 assets/convert_all_v2_usd.py

# Single-object sim grasp test
sim45 sim/run_grasp.py --hdf5 output/grasps/O02_0010_00003_grasp.hdf5
```

---

## Key Paths

| Directory | Contents |
|---|---|
| `~/Project/OakInk2/OakInk-v2-hub/` | OakInk-v2 raw data |
| `~/Project/mano_v1_2/` | MANO hand models |
| `output/contacts_v2/` | Extracted v2 contact maps (NPZ) |
| `output/dataset_v1v2/` | Merged HDF5 training set |
| `output/checkpoints_v1v2/` | Trained model weights |
| `output/grasps/` | Generated grasp pose HDF5 files |

## config.py Key Settings

```python
OAKINK_DIR     = "~/Project/OakInk"           # v1 data
OAKINK2_OBJ_DIR = "~/Project/OakInk2/OakInk-v2-hub/object_raw/align_ds"  # v2 meshes
CONTACTS_V2_DIR = "output/contacts_v2"         # v2 contact map output
```

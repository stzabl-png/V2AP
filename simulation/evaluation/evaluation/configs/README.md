# Evaluation object lists

## Object lists (`tools/build_eval_object_list.py`)

Two sources via `--source`:

| `--source` | Counts |
|------------|--------|
| `robot_gt` (default) | Sum `successful_grasps` in `robot_gt/round_R/` for **R ≥ `--min-round`** |
| `merged` | **Trusted** successes in `merged/{obj}_robot_gt_merged.hdf5` (`gripper_tips_trusted` / at_close; same as affordance prep) |

```bash
# robot_gt, round>=3, success>=30
python tools/build_eval_object_list.py \
  --source robot_gt \
  --outdir output/grasp_collect_no_rot \
  --min-round 3 --min-success 30 \
  --output evaluation/configs/eval_objects_merged_success_ge30.csv

# merged trusted only
python tools/build_eval_object_list.py \
  --source merged \
  --merged-dir output/grasp_collect_no_rot/merged \
  --min-success 40 \
  --output evaluation/configs/eval_objects_merged_success_ge40.csv
```

## Sim USD bundle

Packed assets for external Isaac Sim eval:

```bash
python tools/package_eval_ge30_usd_bundle.py
# → output/eval_objects_ge30_sim_bundle/
```

## Multi-round ablation (`tools/run_round_eval.py`)

Setups **1a–4b** × **10 rounds** (default): each run is full `eval_pool` (candidate gen + sim), 5 trials/object, results under `output/round_eval/{round}_{setup}/`, summary CSV at `output/round_eval/round_eval_summary.csv`.

| ID | HP affordance | Filter | List |
|----|---------------|--------|------|
| 1a/1b | yes | no | GE30 / unseen |
| 2a/2b | no | yes | GE30 / unseen |
| 3a/3b | no | no | GE30 / unseen |
| 4a/4b | yes | yes | GE30 / unseen |

```bash
python tools/run_round_eval.py \
  --candidate-gpu-ids 0,1 --candidate-workers 6 \
  --sim-gpu-ids 0,1 --sim-per-gpu 4
```

Each setup runs `eval_pool` with **`--loud`** by default (`--no-loud` for quiet `--log-only`).

```bash
# Only 2a and 3b
python tools/run_round_eval.py --setups 2a,3b ...

# From 2a through 3b each round (skip 1a,1b)
python tools/run_round_eval.py --start-setup 2a ...

# Round 5 onward, only unseen ablations
python tools/run_round_eval.py --start-round 5 --setups 1b,2b,3b --resume
```

## Unseen sim bundle (`eval_unseen_all.csv`)

```bash
# 1) USD + manifest
python tools/package_eval_unseen_sim_bundle.py

# 2) 500 pool candidates per object (HDF5 only)
python evaluation/eval_pool.py \
  --obj-list evaluation/configs/eval_unseen_all.csv \
  --generate-candidate-each-trial \
  --candidates-only \
  --trials-per-obj-yaw 500 \
  --z-yaw-deg 0 \
  --no-filtering \
  --result-dir output/evaluation/unseen_yaw0_n500_poolgen \
  --candidate-gpu-ids 0,1 \
  --candidate-workers 6

# 3) HDF5 → candidates/{obj_id}.json in bundle
python tools/migrate_eval_unseen_pool_candidates_to_bundle.py
```

## Deprecated lists

- **`eval_objects_merged_success_ge20.csv`** — round≥3, success≥20 (97 objects).
- **`eval_objects_merged_success_ge40.csv`** — regenerate with `--source merged` (trusted) or `--source robot_gt`.

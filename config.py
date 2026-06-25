"""
V2AP — Global Configuration
========================================
All paths and default parameters in one place.
Uses environment variables for external tool paths (no hardcoded machine-specific paths).
"""

import os

# ============================================================
# Project Paths (auto-detected, always portable)
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

# ============================================================
# data/ — Raw Datasets
# ============================================================
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EGO_DATA_DIR = os.path.join(DATA_DIR, "egocentric")   # egodex, hoi4d
TP_DATA_DIR  = os.path.join(DATA_DIR, "third_person")  # dexycb, oakink_v1, ...

# ============================================================
# data_hub/ — Pipeline Intermediate Results
# ============================================================
DATA_HUB = os.path.join(PROJECT_DIR, "data_hub")

MANO_DIR          = os.path.join(DATA_HUB, "mano")
VIDEO_PARAMS_DIR  = os.path.join(DATA_HUB, "video_params")
ROBOT_GT_DIR      = os.path.join(DATA_HUB, "robot_posterior")   # formerly robot_gt
ROBOT_POSTERIOR_DIR = ROBOT_GT_DIR                                # canonical alias

HUMAN_PRIOR_DIR   = os.path.join(DATA_HUB, "human_prior")
# Egocentric object-pose output root. Datasets (hoi4d, egodex) are SIBLINGS under
# here: {EGO_POSE_DIR}/{dataset}/{seq_id}/{ob_in_cam,track_vis}/. Canonical for FP
# writer + downstream readers (contact map, mano align). Replaces legacy
# obj_poses_ego/ and the over-nested poses/Egocentric/Egodex/ layout.
EGO_POSE_DIR      = os.path.join(DATA_HUB, "ProcessedData", "poses", "Egocentric")
MESH_DIR          = os.path.join(DATA_HUB, "meshes")
MESH_RAW_DIR      = os.path.join(MESH_DIR, "raw")
MESH_SCALED_DIR   = os.path.join(MESH_DIR, "scaled")           # SAM3D scaled meshes
MESH_SAM3D_DIR    = os.path.join(MESH_DIR, "SAM3DMesh")

TRAINING_M5_DIR   = os.path.join(DATA_HUB, "training_m5")
TRAINING_M6_DIR   = os.path.join(DATA_HUB, "training_m6")
TRAINING_DIR      = os.path.join(DATA_HUB, "affordance_training_data")
REGISTRY_PATH     = os.path.join(DATA_HUB, "registry.json")

# Legacy mesh aliases (kept for backward compatibility)
MESH_V1_DIR   = os.path.join(MESH_DIR, "v1")
MESH_V2_DIR   = os.path.join(MESH_DIR, "v2")
MESH_CP_DIR   = os.path.join(MESH_DIR, "contactpose")
GRASP_MESH_DIR = os.path.join(MESH_DIR, "grasp_collection")
OAKINK_OBJ_DIR      = MESH_V1_DIR
OAKINK2_OBJ_DIR     = MESH_V2_DIR

# ============================================================
# output/ subdirectories
# ============================================================
CHECKPOINT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
LOGS_DIR         = os.path.join(OUTPUT_DIR, "logs")
VIS_DIR          = os.path.join(OUTPUT_DIR, "vis")
EVAL_DIR         = os.path.join(OUTPUT_DIR, "eval")
CONTACTS_DIR     = os.path.join(OUTPUT_DIR, "contacts")
DATASET_DIR      = os.path.join(OUTPUT_DIR, "dataset_new")
GRASPS_DIR       = os.path.join(EVAL_DIR, "grasps")

# ============================================================
# thirdparty/ — External Tool Paths
# Only needed for upstream data generation, NOT for training/inference.
# ============================================================
THIRDPARTY_DIR = os.path.join(PROJECT_DIR, "thirdparty")
ISAAC_SIM_PATH = os.environ.get("ISAAC_SIM_PATH", "")

HAWOR_DIR  = os.environ.get("HAWOR_DIR",  os.path.join(THIRDPARTY_DIR, "hawor"))
HAPTIC_DIR = os.environ.get("HAPTIC_DIR", os.path.join(THIRDPARTY_DIR, "haptic"))
HAPTIC_MANO_DIR = os.environ.get(
    "HAPTIC_MANO_DIR",
    os.path.join(HAPTIC_DIR, "assets", "mano")
)

ARCTIC_ROOT = os.environ.get("ARCTIC_ROOT", "")
MANO_MODELS = os.path.join(ARCTIC_ROOT, "mano_v1_2", "models") if ARCTIC_ROOT else ""

# FoundationPose
FP_ROOT = os.environ.get("FP_ROOT", os.path.join(THIRDPARTY_DIR, "foundationpose"))

# DepthPro (formerly ml-depth-pro)
DEPTHPRO_DIR = os.path.join(THIRDPARTY_DIR, "depthpro")

# SAM3D mesh reconstruction
SAM3D_DIR     = os.environ.get("SAM3D_DIR", os.path.join(THIRDPARTY_DIR, "sam3d"))
SAM3D_CACHE   = os.path.join(OUTPUT_DIR, "sam3d_obj_cache")
SAM3D_PLY_CACHE = os.path.join(OUTPUT_DIR, "sam3d_mesh_cache")
SAM3D_USER    = os.environ.get("SAM3D_USER", os.environ.get("USER", "user"))

# SAM2 interactive segmentation (installed at /home/lyh/Project/sam2)
SAM2_DIR = os.environ.get("SAM2_DIR", os.path.join(os.path.dirname(PROJECT_DIR), "sam2"))

# MegaSAM camera tracking (formerly mega-sam/)
MEGASAM_DIR    = os.path.join(THIRDPARTY_DIR, "megasam")
MEGASAM_OUTPUT = os.path.join(MEGASAM_DIR, "outputs")
MEGASAM_RECON  = os.path.join(MEGASAM_DIR, "reconstructions")

# OakInk annotation repo
OAKINK_ANNO_DIR = os.environ.get("OAKINK_ANNO_DIR", "")

# ContactPose (optional external dataset)
CONTACTPOSE_DIR = os.environ.get("CONTACTPOSE_DIR", "")
CONTACTPOSE_DATA_DIR = os.path.join(
    CONTACTPOSE_DIR, "ContactPose sample data", "contactpose_data"
) if CONTACTPOSE_DIR else ""

# MANO/Contact caches (in logs/)
HAWOR_CACHE    = os.path.join(LOGS_DIR, "hawor_arctic_cache")
HAPTIC_CACHE   = os.path.join(LOGS_DIR, "haptic_arctic_cache")
ONSET_JSON     = os.path.join(LOGS_DIR, "arctic_grasp_onset.json")
CONTACT_VIS_DIR = os.path.join(VIS_DIR, "contact_region_vis")


# ============================================================
# Default Parameters
# ============================================================

# Contact extraction
CONTACT_THRESHOLD = 0.005    # 5mm contact distance threshold
FRAME_STEP = 5               # Sample every N frames

# Video contact thresholds
GT_CONTACT_TH = 0.015        # 15mm — GT true contact
PRED_CONTACT_TH = 0.030      # 30mm — prediction threshold

# Stability filtering
MIN_FINGERS = 3              # Min fingers in simultaneous contact
MIN_STABLE_FRAMES = 10       # Min consecutive stable frames

# Dataset
NUM_POINTS = 1024            # Point cloud sample count
CONTACT_RADIUS = 0.005       # Contact label radius

# Training
TRAIN_EPOCHS = 150
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 0.001

# Inference
AFFORDANCE_THRESHOLD = 0.3   # Contact probability threshold

# Simulation
OBJECT_SCALE = 1.5
TABLE_TOP_Z = 0.80
ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]

# Robot GT
GAUSSIAN_SIGMA = 0.005       # 5mm gaussian kernel radius

# ============================================================
# Utilities
# ============================================================
def ensure_dirs():
    """Create all output directories."""
    for d in [CONTACTS_DIR, DATASET_DIR, CHECKPOINT_DIR, GRASPS_DIR,
              LOGS_DIR, VIS_DIR, EVAL_DIR,
              HUMAN_PRIOR_DIR, ROBOT_POSTERIOR_DIR, TRAINING_DIR, TRAINING_M5_DIR,
              GRASP_MESH_DIR, HAWOR_CACHE, HAPTIC_CACHE, CONTACT_VIS_DIR]:
        os.makedirs(d, exist_ok=True)

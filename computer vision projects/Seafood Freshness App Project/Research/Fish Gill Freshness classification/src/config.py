"""
config.py

Project configuration for Fish Gill Classification
"""

from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =====================================================
# DATA PATHS
# =====================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
SPLITS_DIR = DATA_DIR / "splits"

# =====================================================
# MODEL PATHS
# =====================================================

MODELS_DIR = PROJECT_ROOT / "models"

BEST_MODEL_NAME = (
    "model_01_Logistic_Regression_features_raw_pixels.pkl"
)

BEST_MODEL_PATH = MODELS_DIR / BEST_MODEL_NAME

# =====================================================
# RESULTS
# =====================================================

RESULTS_DIR = PROJECT_ROOT / "results"

# =====================================================
# PREPROCESSING SETTINGS
# =====================================================

TARGET_SIZE = (360, 360)

NORMALIZATION = "minmax"

USE_CLAHE = True

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (8, 8)

USE_DENOISING = True

DENOISING_METHOD = "bilateral"

# =====================================================
# CLASS LABELS
# =====================================================

HEALTHY_LABEL = 1
UNHEALTHY_LABEL = 0

CLASS_NAMES = {
    0: "Unhealthy",
    1: "Healthy"
}

# =====================================================
# DEPLOYMENT SETTINGS
# =====================================================

DEFAULT_THRESHOLD = 0.60

HIGH_CONFIDENCE_POSITIVE = 0.70

HIGH_CONFIDENCE_NEGATIVE = 0.30

MANUAL_REVIEW_LOWER = 0.40

MANUAL_REVIEW_UPPER = 0.70

# =====================================================
# RANDOMNESS
# =====================================================

RANDOM_STATE = 42

# =====================================================
# SPLITS
# =====================================================

# Dataset paths

GOOD_PATH = DATA_DIR / "good"
BAD_PATH = DATA_DIR / "bad"

# Split configuration

TEST_SIZE = 0.20

# 20% of train_val
# Gives:
# Train ≈ 64%
# Val ≈ 16%
# Test = 20%

VAL_SIZE = 0.20
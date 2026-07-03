"""
Configuration for Fish Segmentation Project
"""

# =============================
# Dataset
# =============================

DATASET_CONFIG = "data.yaml"

# =============================
# Model
# =============================

MODEL_NAME = "yolov8n-seg.pt"

# =============================
# Training
# =============================

EPOCHS = 100

IMAGE_SIZE = 640

BATCH_SIZE = 16

DEVICE = 0      # GPU
# DEVICE = "cpu"

WORKERS = 8

PRETRAINED = True

PATIENCE = 20

SAVE = True

VERBOSE = True

# =============================
# Output
# =============================

PROJECT_NAME = "results"

RUN_NAME = "fish_segmentation"

# =============================
# Optimization
# =============================

OPTIMIZER = "auto"

LEARNING_RATE = 0.01

WEIGHT_DECAY = 0.0005

MOMENTUM = 0.937

# =============================
# Augmentation
# =============================

HSV_H = 0.015

HSV_S = 0.7

HSV_V = 0.4

FLIP_LR = 0.5

FLIP_UD = 0.0

MOSAIC = 1.0

MIXUP = 0.0
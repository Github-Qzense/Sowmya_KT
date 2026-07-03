"""
train_model.py

Simple training script for reproducing the
final Fish Gill Classification model.

Final model:
Logistic Regression + Raw Pixels
"""

from pathlib import Path
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.config import *
from src.logger import get_logger

logger = get_logger(__name__)

def load_data():
    """
    Load train/validation/test splits.
    """
    logger.info("Loading training data")
    
    X_train = np.load(
        SPLITS_DIR / "X_train.npy"
    )

    y_train = np.load(
        SPLITS_DIR / "y_train.npy"
    )

    X_val = np.load(
        SPLITS_DIR / "X_val.npy"
    )

    y_val = np.load(
        SPLITS_DIR / "y_val.npy"
    )

    X_test = np.load(
        SPLITS_DIR / "X_test.npy"
    )

    y_test = np.load(
        SPLITS_DIR / "y_test.npy"
    )

    logger.info(f"Training samples: {len(X_train)}")
    
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


def extract_raw_pixels(images):
    """
    Convert images into flattened vectors.

    (N,360,360,3)
        →
    (N,388800)
    """

    return images.reshape(images.shape[0], -1)


def train():
    """
    Train final production model.
    """

    print("=" * 60)
    print("FISH GILL CLASSIFIER TRAINING")
    print("=" * 60)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    # ==================================================
    # RAW PIXEL FEATURES
    # ==================================================

    X_train = extract_raw_pixels(X_train)
    X_val = extract_raw_pixels(X_val)
    X_test = extract_raw_pixels(X_test)

    print("\nFeature Shape:")
    print(X_train.shape)

    # ==================================================
    # SCALING
    # ==================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ==================================================
    # MODEL
    # ==================================================

    logger.info("Training Logistic Regression model")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    print("\nTraining model...")

    model.fit(
        X_train_scaled,
        y_train
    )

    # ==================================================
    # EVALUATION
    # ==================================================

    train_acc = accuracy_score(
        y_train,
        model.predict(X_train_scaled)
    )

    val_acc = accuracy_score(
        y_val,
        model.predict(X_val_scaled)
    )

    test_acc = accuracy_score(
        y_test,
        model.predict(X_test_scaled)
    )

    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    
    print("\nResults")
    print("-" * 30)

    print(
        f"Train Accuracy : {train_acc:.4f}"
    )

    print(
        f"Val Accuracy   : {val_acc:.4f}"
    )

    print(
        f"Test Accuracy  : {test_acc:.4f}"
    )

    # ==================================================
    # SAVE MODEL
    # ==================================================

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_package = {
        "model": model,
        "scaler": scaler,
        "feature_method": "raw_pixels",
        "model_name": "Logistic Regression",
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }

    save_path = (
        MODELS_DIR
        / "fish_gill_classifier.pkl"
    )

    joblib.dump(
        model_package,
        save_path
    )
    
    logger.info(f"Model saved to {save_path}")
    print("\nModel saved:")
    print(save_path)

    return model_package


if __name__ == "__main__":
    train()
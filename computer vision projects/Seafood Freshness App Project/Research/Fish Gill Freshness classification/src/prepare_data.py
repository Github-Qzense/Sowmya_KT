"""
prepare_data.py

Prepare dataset for Fish Gill Classification.

Pipeline:
Raw Images
    ↓
Preprocessing
    ↓
Train/Val/Test Split
    ↓
Save NPY Files
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import PreprocessingPipeline
from src.config import (
    GOOD_PATH,
    BAD_PATH,
    SPLITS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE
)


def load_image_paths(directory):
    """
    Load all image paths from a directory.
    """

    directory = Path(directory)

    image_paths = []

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(directory.glob(ext))

    return sorted(image_paths)


def process_class_images(
    image_paths,
    label,
    pipeline
):
    """
    Process all images belonging to one class.
    """

    images = []
    labels = []

    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):

        try:
            image = pipeline.preprocess_image(
                image_path
            )

            images.append(image)
            labels.append(label)

            if idx % 25 == 0:
                print(
                    f"Processed {idx}/{total}"
                )

        except Exception as e:

            print(
                f"Skipped {image_path.name}: {e}"
            )

    return images, labels


def create_splits(X, y):
    """
    Create train/val/test splits.
    """

    # Test split

    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    # Validation split

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train_val
        )
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def save_splits(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """
    Save all split files.
    """

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        SPLITS_DIR / "X_train.npy",
        X_train
    )

    np.save(
        SPLITS_DIR / "X_val.npy",
        X_val
    )

    np.save(
        SPLITS_DIR / "X_test.npy",
        X_test
    )

    np.save(
        SPLITS_DIR / "y_train.npy",
        y_train
    )

    np.save(
        SPLITS_DIR / "y_val.npy",
        y_val
    )

    np.save(
        SPLITS_DIR / "y_test.npy",
        y_test
    )


def main():

    print("=" * 60)
    print("FISH GILL DATA PREPARATION")
    print("=" * 60)

    pipeline = PreprocessingPipeline()

    # ==================================================
    # LOAD PATHS
    # ==================================================

    healthy_paths = load_image_paths(
        GOOD_PATH
    )

    unhealthy_paths = load_image_paths(
        BAD_PATH
    )

    print(
        f"Healthy Images: {len(healthy_paths)}"
    )

    print(
        f"Unhealthy Images: {len(unhealthy_paths)}"
    )

    # ==================================================
    # PROCESS HEALTHY
    # ==================================================

    print("\nProcessing Healthy Images...")

    healthy_images, healthy_labels = (
        process_class_images(
            healthy_paths,
            label=1,
            pipeline=pipeline
        )
    )

    # ==================================================
    # PROCESS UNHEALTHY
    # ==================================================

    print("\nProcessing Unhealthy Images...")

    unhealthy_images, unhealthy_labels = (
        process_class_images(
            unhealthy_paths,
            label=0,
            pipeline=pipeline
        )
    )

    # ==================================================
    # COMBINE
    # ==================================================

    X = np.array(
        healthy_images + unhealthy_images,
        dtype=np.float32
    )

    y = np.array(
        healthy_labels + unhealthy_labels,
        dtype=np.int32
    )

    print("\nDataset Summary")
    print("-" * 30)

    print(
        f"Total Samples: {len(X)}"
    )

    print(
        f"Healthy: {np.sum(y == 1)}"
    )

    print(
        f"Unhealthy: {np.sum(y == 0)}"
    )

    # ==================================================
    # SPLITS
    # ==================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = create_splits(X, y)

    print("\nSplit Summary")
    print("-" * 30)

    print(
        f"Train: {len(X_train)}"
    )

    print(
        f"Val:   {len(X_val)}"
    )

    print(
        f"Test:  {len(X_test)}"
    )

    # ==================================================
    # SAVE
    # ==================================================

    save_splits(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print("\nSaved split files:")
    print(SPLITS_DIR)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
"""
utils.py

Utility functions for Fish Gill Classification Project
"""

import json
import joblib
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any


# =====================================================
# DIRECTORY UTILITIES
# =====================================================

def create_directory(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =====================================================
# JSON UTILITIES
# =====================================================

def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary as JSON file.
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load JSON file.
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        return json.load(f)


# =====================================================
# MODEL UTILITIES
# =====================================================

def load_model(model_path: Union[str, Path]) -> Dict:
    """
    Load saved model package.

    Expected format:
    {
        'model': model,
        'scaler': scaler,
        'feature_method': feature_name,
        'model_name': model_name,
        ...
    }
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    return joblib.load(model_path)


def save_model(model_object: Any,
               filepath: Union[str, Path]) -> None:
    """
    Save model using joblib.
    """
    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model_object, filepath)


# =====================================================
# IMAGE UTILITIES
# =====================================================

def load_image(
    image_path: Union[str, Path],
    rgb: bool = True
) -> np.ndarray:
    """
    Load image from disk.

    Args:
        image_path: image file path
        rgb: convert BGR → RGB

    Returns:
        numpy image array
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Failed to read image:\n{image_path}"
        )

    if rgb:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    return image


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    rgb: bool = True
) -> None:
    """
    Save image to disk.

    Args:
        image: image array
        output_path: save location
        rgb: convert RGB → BGR before saving
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_img = image.copy()

    if rgb:
        save_img = cv2.cvtColor(
            save_img,
            cv2.COLOR_RGB2BGR
        )

    cv2.imwrite(str(output_path), save_img)


# =====================================================
# IMAGE VALIDATION
# =====================================================

def is_valid_image(
    image: np.ndarray,
    min_size: int = 100,
    max_size: int = 2000
) -> bool:
    """
    Validate image dimensions.

    Args:
        image: numpy image
        min_size: minimum height/width
        max_size: maximum height/width

    Returns:
        bool
    """

    if image is None:
        return False

    if len(image.shape) != 3:
        return False

    if image.shape[2] != 3:
        return False

    h, w = image.shape[:2]

    if h < min_size or w < min_size:
        return False

    if h > max_size or w > max_size:
        return False

    return True


# =====================================================
# DATASET UTILITIES
# =====================================================

def load_image_paths(
    directory: Union[str, Path],
    extensions=(".png", ".jpg", ".jpeg")
):
    """
    Load image paths from directory.

    Args:
        directory: folder containing images

    Returns:
        sorted list of image paths
    """

    directory = Path(directory)

    image_paths = []

    for ext in extensions:
        image_paths.extend(
            directory.glob(f"*{ext}")
        )

    return sorted(image_paths)


# =====================================================
# REPORTING UTILITIES
# =====================================================

def print_separator(length: int = 60):
    """
    Pretty separator line.
    """
    print("=" * length)


def print_section(title: str):
    """
    Pretty section header.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# =====================================================
# NUMPY HELPERS
# =====================================================

def image_stats(image: np.ndarray) -> Dict:
    """
    Compute image statistics.
    """

    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
    }
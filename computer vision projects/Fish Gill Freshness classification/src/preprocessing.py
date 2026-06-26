"""
preprocessing.py

Image preprocessing pipeline for Fish Gill Classification
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from src.config import (
    TARGET_SIZE,
    NORMALIZATION,
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    USE_DENOISING,
    DENOISING_METHOD
)

from src.utils import load_image, is_valid_image
from src.logger import get_logger

logger = get_logger(__name__)

class PreprocessingPipeline:
    """
    Fish gill image preprocessing pipeline.
    """

    def __init__(
        self,
        target_size=TARGET_SIZE,
        normalization=NORMALIZATION
    ):
        self.target_size = target_size
        self.normalization = normalization

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate image dimensions and format.
        """
        return is_valid_image(image)

    # =====================================================
    # RESIZING
    # =====================================================

    def resize_image(
        self,
        image: np.ndarray,
        interpolation=cv2.INTER_AREA
    ) -> np.ndarray:
        """
        Resize image.
        """
        return cv2.resize(
            image,
            self.target_size,
            interpolation=interpolation
        )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Normalize image.
        """

        image = image.astype(np.float32)

        if self.normalization == "minmax":
            image = image / 255.0

        elif self.normalization == "standard":
            output = np.zeros_like(image)

            for channel in range(3):
                mean = image[:, :, channel].mean()
                std = image[:, :, channel].std()

                output[:, :, channel] = (
                    image[:, :, channel] - mean
                ) / (std + 1e-7)

            image = output

        elif self.normalization == "none":
            pass

        return image

    # =====================================================
    # CLAHE
    # =====================================================

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE
    ) -> np.ndarray:
        """
        Apply CLAHE enhancement.
        """

        image_uint8 = (image * 255).astype(np.uint8)

        lab = cv2.cvtColor(
            image_uint8,
            cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        l = clahe.apply(l)

        merged = cv2.merge([l, a, b])

        enhanced = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2RGB
        )

        return enhanced.astype(np.float32) / 255.0

    # =====================================================
    # DENOISING
    # =====================================================

    def remove_noise(
        self,
        image: np.ndarray,
        method=DENOISING_METHOD
    ) -> np.ndarray:
        """
        Remove image noise.
        """

        if image.max() <= 1:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        if method == "bilateral":

            denoised = cv2.bilateralFilter(
                image_uint8,
                d=9,
                sigmaColor=75,
                sigmaSpace=75
            )

        elif method == "gaussian":

            denoised = cv2.GaussianBlur(
                image_uint8,
                (5, 5),
                0
            )

        elif method == "median":

            denoised = cv2.medianBlur(
                image_uint8,
                5
            )

        else:
            denoised = image_uint8

        return denoised.astype(np.float32) / 255.0

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def preprocess(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Steps:
            Validation
            Denoising
            CLAHE
            Resize
            Normalization

        Returns:
            Processed image
        """

        if not self.validate_image(image):
            raise ValueError(
                "Invalid image format or dimensions."
            )

        image = image.astype(np.float32) / 255.0

        if USE_DENOISING:
            image = self.remove_noise(
                image,
                method=DENOISING_METHOD
            )

        if USE_CLAHE:
            image = self.apply_clahe(
                image,
                clip_limit=CLAHE_CLIP_LIMIT,
                tile_grid_size=CLAHE_TILE_GRID_SIZE
            )

        image = self.resize_image(image)

        image = self.normalize_image(
            (image * 255).astype(np.uint8)
        )

        return image

    # =====================================================
    # FILE INTERFACE
    # =====================================================

    def preprocess_image(
        self,
        image_path: str
    ) -> np.ndarray:
        """
        Load and preprocess image directly from path.
        """
        logger.info(f"Processing image: {image_path}")
        image = load_image(image_path)

        return self.preprocess(image)

    # =====================================================
    # METADATA
    # =====================================================

    def get_metadata(self) -> Dict:
        """
        Return preprocessing settings.
        """

        return {
            "target_size": self.target_size,
            "normalization": self.normalization,
            "clahe": USE_CLAHE,
            "denoising": USE_DENOISING,
            "denoising_method": DENOISING_METHOD,
        }
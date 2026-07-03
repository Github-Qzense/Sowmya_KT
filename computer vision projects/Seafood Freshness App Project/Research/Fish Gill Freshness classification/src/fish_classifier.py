"""
fish_classifier.py

Main Fish Gill Classification class.

Pipeline:
Image
    ↓
Preprocessing
    ↓
Raw Pixel Extraction
    ↓
Scaler
    ↓
Logistic Regression
    ↓
Prediction
"""

import numpy as np
from pathlib import Path

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEFAULT_THRESHOLD,
    HIGH_CONFIDENCE_POSITIVE,
    HIGH_CONFIDENCE_NEGATIVE,
    MANUAL_REVIEW_LOWER,
    MANUAL_REVIEW_UPPER
)

from src.utils import load_model
from src.preprocessing import PreprocessingPipeline


class FishGillClassifier:
    """
    Fish Gill Disease Classification System
    """

    def __init__(
        self,
        model_path=BEST_MODEL_PATH
    ):
        self.model_path = Path(model_path)

        self.pipeline = PreprocessingPipeline()

        self.model = None
        self.scaler = None

        self.model_name = None
        self.feature_method = None

        self.load_model()

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def load_model(self):
        """
        Load saved model package.
        """

        package = load_model(self.model_path)

        self.model = package["model"]
        self.scaler = package["scaler"]

        self.model_name = package.get(
            "model_name",
            "Unknown"
        )

        self.feature_method = package.get(
            "feature_method",
            "raw_pixels"
        )

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    def extract_features(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Raw pixel feature extraction.

        Converts:
        (360,360,3)

        Into:

        (1,388800)
        """

        features = image.reshape(1, -1)

        return features

    # =====================================================
    # FEATURE PROCESSING
    # =====================================================

    def prepare_features(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Full feature preparation pipeline.
        """

        features = self.extract_features(image)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        return features

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(
        self,
        image: np.ndarray
    ) -> int:
        """
        Predict class label.

        Returns:
            0 -> Unhealthy
            1 -> Healthy
        """

        features = self.prepare_features(image)

        prediction = self.model.predict(features)[0]

        return int(prediction)

    # =====================================================
    # PROBABILITY
    # =====================================================

    def predict_proba(
        self,
        image: np.ndarray
    ) -> float:
        """
        Predict probability of HEALTHY class.

        Returns:
            float between 0 and 1
        """

        features = self.prepare_features(image)

        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(
                features
            )[0, 1]

        else:
            prediction = self.model.predict(features)[0]
            probability = float(prediction)

        return float(probability)

    # =====================================================
    # CONFIDENCE
    # =====================================================

    def get_confidence(
        self,
        probability: float
    ) -> float:
        """
        Convert probability into confidence score.
        """

        return max(
            probability,
            1 - probability
        )

    # =====================================================
    # REVIEW STATUS
    # =====================================================

    def get_review_status(
        self,
        probability: float
    ) -> str:
        """
        Determine review recommendation.
        """

        if (
            probability >= HIGH_CONFIDENCE_POSITIVE
            or
            probability <= HIGH_CONFIDENCE_NEGATIVE
        ):
            return "Automatic Approval"

        if (
            MANUAL_REVIEW_LOWER
            <= probability
            <= MANUAL_REVIEW_UPPER
        ):
            return "Manual Review Required"

        return "Review Recommended"

    # =====================================================
    # COMPLETE INFERENCE
    # =====================================================

    def predict_image(
        self,
        image_path: str
    ) -> dict:
        """
        Complete prediction pipeline.

        Returns:
            Dictionary with results.
        """

        image = self.pipeline.preprocess_image(
            image_path
        )
        print("\nDEBUG")
        print("Image:", image_path)
        print("Shape:", image.shape)
        print("Min:", image.min())
        print("Max:", image.max())
        print("Mean:", image.mean())
        
        prediction = self.predict(image)

        probability = self.predict_proba(image)

        confidence = self.get_confidence(
            probability
        )

        review_status = self.get_review_status(
            probability
        )

        result = {
            "image_path": str(image_path),

            "prediction": prediction,

            "class_name": CLASS_NAMES[
                prediction
            ],

            "healthy_probability":
                round(probability, 4),

            "unhealthy_probability":
                round(1 - probability, 4),

            "confidence":
                round(confidence, 4),

            "review_status":
                review_status,

            "model":
                self.model_name,

            "features":
                self.feature_method
        }
        print(result)
        return result

    # =====================================================
    # BATCH PREDICTION
    # =====================================================

    def predict_batch(
        self,
        image_paths
    ):
        """
        Predict multiple images.
        """

        results = []

        for image_path in image_paths:
            try:
                results.append(
                    self.predict_image(
                        image_path
                    )
                )

            except Exception as e:

                results.append({
                    "image_path":
                        str(image_path),

                    "error":
                        str(e)
                })

        return results

    # =====================================================
    # MODEL INFO
    # =====================================================

    def get_model_info(self):
        """
        Return model metadata.
        """

        return {
            "model_name":
                self.model_name,

            "feature_method":
                self.feature_method,

            "threshold":
                DEFAULT_THRESHOLD
        }
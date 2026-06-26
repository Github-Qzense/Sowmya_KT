"""
predictor.py

High-level prediction utilities for
Fish Gill Classification.
"""

from pathlib import Path
import pandas as pd

from src.fish_classifier import FishGillClassifier
from src.logger import get_logger

logger = get_logger(__name__)

class FishGillPredictor:
    """
    User-facing prediction interface.
    """

    def __init__(self, model_path=None):
        self.classifier = FishGillClassifier(
            model_path=model_path
        ) if model_path else FishGillClassifier()

    # =====================================================
    # SINGLE IMAGE
    # =====================================================

    def predict_image(
        self,
        image_path
    ):
        """
        Predict a single image.

        Returns:
            dict
        """
        logger.info(f"Predicting image: {image_path}")
        return self.classifier.predict_image(
            image_path
        )

    # =====================================================
    # MULTIPLE IMAGES
    # =====================================================

    def predict_images(
        self,
        image_paths
    ):
        """
        Predict multiple images.

        Returns:
            list of dictionaries
        """

        return self.classifier.predict_batch(
            image_paths
        )

    # =====================================================
    # FOLDER PREDICTION
    # =====================================================

    def predict_folder(
        self,
        folder_path,
        extensions=(".png", ".jpg", ".jpeg")
    ):
        """
        Predict all images in a folder.

        Returns:
            pandas DataFrame
        """

        folder_path = Path(folder_path)

        image_paths = []

        for ext in extensions:
            image_paths.extend(
                folder_path.glob(f"*{ext}")
            )

        image_paths = sorted(image_paths)

        results = self.classifier.predict_batch(
            image_paths
        )

        return pd.DataFrame(results)

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    def save_predictions(
        self,
        dataframe,
        output_csv
    ):
        """
        Save predictions to CSV.
        """

        dataframe.to_csv(
            output_csv,
            index=False
        )

        print(
            f"Predictions saved to:\n{output_csv}"
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def prediction_summary(
        self,
        dataframe
    ):
        """
        Print prediction statistics.
        """

        if dataframe.empty:
            print("No predictions available.")
            return

        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)

        if "class_name" in dataframe.columns:

            counts = (
                dataframe["class_name"]
                .value_counts()
                .to_dict()
            )

            total = len(dataframe)

            for cls, count in counts.items():

                pct = (
                    count / total
                ) * 100

                print(
                    f"{cls:<15}: "
                    f"{count:>5} "
                    f"({pct:.1f}%)"
                )

        if "confidence" in dataframe.columns:

            print(
                f"\nAverage Confidence: "
                f"{dataframe['confidence'].mean():.3f}"
            )

        print(
            f"Total Samples: "
            f"{len(dataframe)}"
        )
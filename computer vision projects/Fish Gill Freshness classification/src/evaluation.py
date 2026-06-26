"""
evaluation.py

Evaluate trained Fish Gill Classification model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    SPLITS_DIR,
    RESULTS_DIR,
    BEST_MODEL_PATH
)

from src.logger import get_logger
logger = get_logger(__name__)

class ModelEvaluator:

    def __init__(
        self,
        model_path=BEST_MODEL_PATH
    ):
        self.model_path = Path(model_path)

        self.model_package = joblib.load(
            self.model_path
        )

        self.model = self.model_package["model"]
        self.scaler = self.model_package["scaler"]

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    @staticmethod
    def extract_features(images):
        """
        Raw pixel extraction.
        """

        return images.reshape(
            images.shape[0],
            -1
        )

    # =====================================================
    # LOAD TEST DATA
    # =====================================================

    def load_test_data(self):

        X_test = np.load(
            SPLITS_DIR / "X_test.npy"
        )

        y_test = np.load(
            SPLITS_DIR / "y_test.npy"
        )

        return X_test, y_test

    # =====================================================
    # PREPARE DATA
    # =====================================================

    def prepare_data(self, X):

        X = self.extract_features(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(self):

        X_test, y_test = (
            self.load_test_data()
        )

        X_test = self.prepare_data(
            X_test
        )

        y_pred = self.model.predict(
            X_test
        )

        if hasattr(
            self.model,
            "predict_proba"
        ):
            y_proba = (
                self.model
                .predict_proba(X_test)[:, 1]
            )

            auc_score = roc_auc_score(
                y_test,
                y_proba
            )

        else:
            auc_score = None

        metrics = {
            "accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "f1_score":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "auc":
                auc_score
        }

        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return (
            metrics,
            y_test,
            y_pred
        )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Unhealthy",
                "Healthy"
            ],
            yticklabels=[
                "Unhealthy",
                "Healthy"
            ]
        )

        plt.title(
            "Confusion Matrix"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        RESULTS_DIR.mkdir(
            exist_ok=True,
            parents=True
        )

        plt.savefig(
            RESULTS_DIR /
            "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================
    # REPORT
    # =====================================================

    def save_report(
        self,
        y_true,
        y_pred
    ):

        report = classification_report(
            y_true,
            y_pred,
            target_names=[
                "Unhealthy",
                "Healthy"
            ]
        )

        report_path = (
            RESULTS_DIR /
            "classification_report.txt"
        )

        with open(
            report_path,
            "w"
        ) as f:

            f.write(report)

    # =====================================================
    # RUN ALL
    # =====================================================

    def run(self):
        logger.info("Running evaluation")
        metrics, y_true, y_pred = (
            self.evaluate()
        )

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : "
            f"{metrics['f1_score']:.4f}"
        )

        if metrics["auc"] is not None:

            print(
                f"AUC      : "
                f"{metrics['auc']:.4f}"
            )

        self.plot_confusion_matrix(
            y_true,
            y_pred
        )

        self.save_report(
            y_true,
            y_pred
        )

        print("\nFiles Saved")

        print(
            f"- {RESULTS_DIR}/confusion_matrix.png"
        )

        print(
            f"- {RESULTS_DIR}/classification_report.txt"
        )


if __name__ == "__main__":

    evaluator = ModelEvaluator()

    evaluator.run()
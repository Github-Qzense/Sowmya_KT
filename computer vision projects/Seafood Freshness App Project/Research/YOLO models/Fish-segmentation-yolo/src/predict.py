"""
predict.py

Run inference using a trained YOLO segmentation model.

Usage:
    python predict.py --image path/to/image.jpg or
    python predict.py --image test.jpg --conf 0.4
"""

from pathlib import Path
import argparse
from ultralytics import YOLO

from src.config import PROJECT_NAME, RUN_NAME

# --------------------------------------------------
# Default model path
# --------------------------------------------------

MODEL_PATH = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"


# --------------------------------------------------
# Predictor Class
# --------------------------------------------------

class FishSegmentor:

    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def predict(
        self,
        image_path,
        save=True,
        show=False,
        conf=0.25
    ):
        """
        Perform segmentation on a single image.

        Args:
            image_path (str): Path to input image
            save (bool): Save prediction image
            show (bool): Display prediction
            conf (float): Confidence threshold

        Returns:
            results
        """

        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save,
            show=show,
            verbose=False
        )

        return results


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Confidence threshold"
    )

    args = parser.parse_args()

    predictor = FishSegmentor()

    results = predictor.predict(
        args.image,
        conf=args.conf
    )

    result = results[0]

    print("=" * 60)
    print("Prediction Summary")
    print("=" * 60)

    print(f"Image: {args.image}")

    print(f"Detected Objects : {len(result.boxes)}")

    if result.boxes is not None:

        for i, box in enumerate(result.boxes):

            cls = int(box.cls)

            conf = float(box.conf)

            print(
                f"{i+1}. Class={result.names[cls]}  Confidence={conf:.3f}"
            )

    print()
    print("Prediction image saved to:")
    print(result.save_dir)


if __name__ == "__main__":
    main()
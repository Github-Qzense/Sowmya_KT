"""
Train YOLO Segmentation Model
"""

from ultralytics import YOLO

from src.config import *


def main():

    print("=" * 60)
    print("Fish Segmentation Training")
    print("=" * 60)

    print(f"Model      : {MODEL_NAME}")
    print(f"Dataset    : {DATASET_CONFIG}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch Size : {BATCH_SIZE}")
    print(f"Image Size : {IMAGE_SIZE}")
    print(f"Device      : {DEVICE}")
    print()

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATASET_CONFIG,

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        pretrained=PRETRAINED,

        optimizer=OPTIMIZER,

        lr0=LEARNING_RATE,

        momentum=MOMENTUM,

        weight_decay=WEIGHT_DECAY,

        patience=PATIENCE,

        hsv_h=HSV_H,

        hsv_s=HSV_S,

        hsv_v=HSV_V,

        fliplr=FLIP_LR,

        flipud=FLIP_UD,

        mosaic=MOSAIC,

        mixup=MIXUP,

        save=SAVE,

        verbose=VERBOSE,

        project=PROJECT_NAME,

        name=RUN_NAME,
    )

    print("\nTraining Finished")
    print(f"Best Model : {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
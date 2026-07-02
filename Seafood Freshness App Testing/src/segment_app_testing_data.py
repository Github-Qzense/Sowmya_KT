"""
segment_app_testing_data.py

Segment fish in app testing images using a YOLO segmentation model
and save cropped instance segments into a new folder with the same
date/species/label structure.

Requirements
------------
1. pip install ultralytics opencv-python tqdm
2. Have a trained YOLO fish segmentation model (e.g. fish.pt).
3. Source folder structure (example):
   src_root/
     2024-12-03/
       sardine/
         Good/
           2024-12-03_10_14_57_(17680)_sardine_input.jpeg
         Bad/
           ...
4. Output will mirror structure under dest_root, with filenames
   suffixed by _(<instance_index>).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


@dataclass
class SegmentationConfig:
    """
    Configuration for segmentation model and paths.
    """
    model_path: str
    src_root: str
    dest_root: str
    file_extension: str = ".jpg"  # output image extension
    imgsz: int = 640
    conf: float = 0.75
    iou: float = 0.7
    white_background: bool = False


def load_segmentation_model(model_path: str) -> YOLO:
    """
    Load a YOLO segmentation model from the given path.
    """
    model = YOLO(model_path)
    return model


def get_segmented_img(
    pred,
    white_background: bool = False,
) -> List[np.ndarray]:
    """
    Extract instance-segmented crops from a YOLO prediction result.

    Parameters
    ----------
    pred : ultralytics.engine.results.Results
        Single result object from YOLO prediction.
    white_background : bool, optional
        If True, replace black background with white in each segment.

    Returns
    -------
    list of np.ndarray
        List of segmented instance images (RGB).
    """
    img = pred.orig_img.copy()
    height, width = pred.orig_shape[0], pred.orig_shape[1]

    if pred.masks is None or pred.masks.xy is None:
        return []

    masks = pred.masks.xy
    seg_img_list: List[np.ndarray] = []

    for mask_points in masks:
        mask_points = mask_points.astype(int)
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [mask_points], 255)

        masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
        x, y, w, h = cv2.boundingRect(mask_points)
        x1, y1, x2, y2 = (x, y, x + w, y + h)

        segment_image = np.zeros((height, width, 3), dtype=np.uint8)
        segment_image[y : y + h, x : x + w] = masked_img[y : y + h, x : x + w]
        segment_image = segment_image[y : y + h, x : x + w]

        if white_background:
            black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
            segment_image[black_mask] = [255, 255, 255]

        seg_img_list.append(segment_image)

    return seg_img_list


def segment_img(
    img_path: str,
    model: YOLO,
    imgsz: int = 640,
    conf: float = 0.75,
    iou: float = 0.7,
    white_background: bool = False,
) -> Optional[List[np.ndarray]]:
    """
    Run YOLO segmentation on a single image and return list of segments (RGB).
    """
    results = model(img_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    res = results[0]

    if res.masks is None:
        return None

    seg_img_list = get_segmented_img(res, white_background=white_background)
    if not seg_img_list:
        return None
    return seg_img_list


def segment_dataset(cfg: SegmentationConfig) -> None:
    """
    Walk through the src_root folder, segment each image, and save
    results under dest_root mirroring the folder structure:

    src_root/date/species/label/*.jpeg
    -> dest_root/date/species/label/*_(i).jpg
    """
    src_root = Path(cfg.src_root)
    dest_root = Path(cfg.dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    model = load_segmentation_model(cfg.model_path)

    # Iterate dates
    for date in sorted(os.listdir(src_root)):
        src_date_path = src_root / date
        if not src_date_path.is_dir():
            continue

        dest_date_path = dest_root / date
        dest_date_path.mkdir(parents=True, exist_ok=True)

        # Iterate species
        for species in tqdm(
            os.listdir(src_date_path),
            desc=f"Processing date {date}",
        ):
            src_species_path = src_date_path / species
            if not src_species_path.is_dir():
                continue

            dest_species_path = dest_date_path / species
            dest_species_path.mkdir(parents=True, exist_ok=True)

            # Iterate labels
            for label in os.listdir(src_species_path):
                src_label_path = src_species_path / label
                if not src_label_path.is_dir():
                    continue

                dest_label_path = dest_species_path / label
                dest_label_path.mkdir(parents=True, exist_ok=True)

                # Iterate images
                for img_name in os.listdir(src_label_path):
                    src_img_path = src_label_path / img_name
                    if not src_img_path.is_file():
                        continue

                    seg_img_list = segment_img(
                        str(src_img_path),
                        model=model,
                        imgsz=cfg.imgsz,
                        conf=cfg.conf,
                        iou=cfg.iou,
                        white_background=cfg.white_background,
                    )
                    if seg_img_list is None:
                        continue

                    base_name = src_img_path.stem
                    for i, seg_img in enumerate(seg_img_list):
                        seg_bgr = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
                        out_name = f"{base_name}_({i}){cfg.file_extension}"
                        out_path = dest_label_path / out_name

                        if out_path.exists():
                            continue

                        cv2.imwrite(str(out_path), seg_bgr)


def main():
    """
    Example usage:
    Configure model path and folders, then run:

        python segment_app_testing_data.py
    """
    cfg = SegmentationConfig(
        model_path="fish.pt", # replace this with the fish segmentation yolo model file path
        src_root=r"downloaded_data\input",
        dest_root="app_testing_data_segmented",
        file_extension=".jpg",
        imgsz=640,
        conf=0.75,
        iou=0.7,
        white_background=False,
    )

    segment_dataset(cfg)
    print("Segmentation completed.")


if __name__ == "__main__":
    main()

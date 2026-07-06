"""
segment_fish.py

Utility to segment a fish from an input image and save it with a white background.
Update the YOLO openvino model path and image path in the code.
run this file by using the command in terminal: python path/to/segment_fish.py
"""

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------------
# Load Fish Segmentation Model
# ------------------------------------------------------------------
FISH_MODEL_PATH = r"../../mobileapp models/best_openvino_model" # replace with your model path

fish_model = YOLO(FISH_MODEL_PATH, task="segment")
print("Model loaded")

# Warm-up
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
fish_model.predict(dummy, verbose=False)
print("Model warmed up")

# ------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------
def extract_single_image_segment(
    image,
    mask_segment,
    shape,
    background="white",
    add_padding = True,
    padding_size = 20,
):
    """
    Extract segmented object using polygon mask.

    Parameters
    ----------
    image : np.ndarray
    mask_segment : ndarray
        Polygon coordinates.
    shape : tuple
        Original image shape.
    background : str
        "white" or "black"
    padding : int

    Returns
    -------
    np.ndarray
    """

    height, width = shape[:2]

    segment = np.array(mask_segment, dtype=np.int32).reshape((-1, 2))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 255)

    masked = cv2.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv2.boundingRect(segment)

    cropped = np.zeros((h, w, 3), dtype=np.uint8)
    cropped[:] = (255, 255, 255) if background.lower() == "white" else (0, 0, 0)

    roi = masked[y:y+h, x:x+w]

    object_pixels = np.any(roi != 0, axis=-1)

    cropped[object_pixels] = roi[object_pixels]
    
    if add_padding:
        # Add padding
        cropped = cv2.copyMakeBorder(
            cropped,
            padding_size,
            padding_size,
            padding_size,
            padding_size,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255) if background.lower() == "white" else (0, 0, 0),
        )

    return cropped


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
def segment_fish(
    image,
    save_path=r"../generated-results/segmented_fish.png",
    conf=0.75,
    dpi=300,
    show=True,
):
    """
    Segment the largest fish from an image.

    Parameters
    ----------
    image : str or np.ndarray
        Image path or RGB image.

    save_path : str
        Output filename.

    conf : float
        Detection confidence.

    dpi : int
        Image clarity.
        150 = screen
        300 = paper
        600 = publication
        1200 = ultra high quality

    show : bool
        Display result.

    Returns
    -------
    segmented_image : np.ndarray
    """

    # ----------------------------------------------------------
    # Read image
    # ----------------------------------------------------------
    if isinstance(image, str):
        filename = (os.path.basename(image)).split(".")[0]
        image = np.array(Image.open(image).convert("RGB"))

    original = image.copy()
    
    # ----------------------------------------------------------
    # Detect fish
    # ----------------------------------------------------------
    results = fish_model.predict(
        original[:, :, ::-1],
        conf=conf,
        verbose=False,
    )

    if len(results[0].boxes) == 0:
        raise ValueError("No fish detected.")

    boxes = results[0].boxes.xyxy.cpu().numpy()
    masks = results[0].masks.xy

    # ----------------------------------------------------------
    # Segment fish
    # ----------------------------------------------------------
    for idx in range(len(boxes)):
        segmented = extract_single_image_segment(
            original,
            masks[idx],
            original.shape,
            background="white",
            add_padding=True,
        )
        print(f"Saving {filename}...")   
        save_path = f"../generated-results/{filename}_segmented_{idx}.png"     
        plt.imsave(save_path, segmented, dpi=dpi)
       
        if show:
            plt.figure(figsize=(6, 6))
            plt.imshow(segmented)
            plt.axis("off")
            plt.title("Segmented Fish")
            plt.show()        
        
        print(f"✓ Saved to: {save_path}")
        print(f"✓ DPI: {dpi}")


# ------------------------------------------------------------------
# Example
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Running example...")
    segmented = segment_fish(
        image=r"../sample-inputs/2024-12-24_10_10_40_(18130)_sardine_input.jpeg", # replace with your image path
        # save_path=r"../generated-results/segmented_fish.png",
        conf=0.75,
        dpi=1200,
    )
    
    segmented = segment_fish(
        image=r"../sample-inputs/2025-01-28_10_50_33_(19200)_mackerel_input.jpeg", # replace with your image path
        # save_path=r"../generated-results/segmented_fish.png",
        conf=0.75,
        dpi=1200,
    )
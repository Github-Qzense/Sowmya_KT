"""
highlight_cuts.py

Detect and highlight cuts/damages on a segmented fish image.
Update the YOLO openvino model path and image path in the code.
run this file by using the command in terminal: python path/to/highlight_cuts.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------------
# Load Cut Segmentation Model
# ------------------------------------------------------------------
# CUT_MODEL_PATH = r"mobileapp models\cut-seg_openvino_model" # replace with your model path
CUT_MODEL_PATH = r"../../mobileapp models/cut-seg_openvino_model" # replace with your model path

cut_model = YOLO(CUT_MODEL_PATH, task="segment")

# Warm-up
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
cut_model.predict(dummy, verbose=False)


# ------------------------------------------------------------------
# Preprocessing (same as your original code)
# ------------------------------------------------------------------
def process_image(image, size=(640, 640)):
    """
    Resize while maintaining aspect ratio and pad with white background.
    Returns resized image and padding details.
    """

    h, w = image.shape[:2]

    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = size[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = size[1]
        new_w = int(new_h * aspect_ratio)

    resized = cv2.resize(image, (new_w, new_h))

    pad_h = (size[1] - new_h) // 2
    pad_w = (size[0] - new_w) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    padded = cv2.resize(padded, (640, 640))

    return padded, {
        "orig_w": w,
        "orig_h": h,
        "new_w": new_w,
        "new_h": new_h,
        "pad_w": pad_w,
        "pad_h": pad_h,
    }

def map_polygon_to_original(polygon, transform):
    """
    Convert polygon coordinates from the 640x640 processed image
    back to the original image.
    """

    polygon = polygon.astype(np.float32)

    polygon[:, 0] -= transform["pad_w"]
    polygon[:, 1] -= transform["pad_h"]

    scale_x = transform["new_w"] / transform["orig_w"]
    scale_y = transform["new_h"] / transform["orig_h"]

    polygon[:, 0] /= scale_x
    polygon[:, 1] /= scale_y

    polygon[:, 0] = np.clip(polygon[:, 0], 0, transform["orig_w"] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, transform["orig_h"] - 1)

    return polygon.astype(np.int32)

# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
def highlight_cuts(
    image,
    save_path=r"../generated-results/fish_cuts.png",
    conf=0.5,
    alpha=0.3,
    dpi=300,
    show=False,
):
    """
    Highlight detected cut/damage regions.

    Parameters
    ----------
    image : str or ndarray
        Segmented fish image (white background).

    save_path : str
        Output filename.

    conf : float
        Detection confidence.

    alpha : float
        Overlay transparency.

    dpi : int
        Saving DPI.

    show : bool
        Display output.

    Returns
    -------
    highlighted_image : ndarray
    """

    # ----------------------------------------------------------
    # Read image
    # ----------------------------------------------------------
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))

    original = image.copy()

    model_input, transform = process_image(original)
    
    results = cut_model.predict(
        model_input[:, :, ::-1],
        conf=conf,
        verbose=False,
    )

    highlighted = original.copy()

    # ----------------------------------------------------------
    # No cuts found
    # ----------------------------------------------------------
    if (
        results[0].masks is None
        or len(results[0].masks.xy) == 0
    ):
        print("✓ No cuts detected.")

    else:
        print(f"✓ {len(results[0].masks.xy)} cuts detected.")
        overlay = highlighted.copy()

        # First loop: only masks
        for polygon in results[0].masks.xy:
            # polygon = np.array(polygon, dtype=np.int32).reshape((-1,1,2))
            
            polygon = map_polygon_to_original(np.array(polygon), transform,)
            polygon = polygon.reshape((-1,1,2))
            cv2.fillPoly(overlay, [polygon], (255,0,0))

        # Blend the blue mask with the original image
        highlighted = cv2.addWeighted(
            overlay,
            alpha,
            highlighted,
            1 - alpha,
            0,
        )
        
        # Second loop: boxes and labels
        for polygon in results[0].masks.xy:
            # polygon = np.array(polygon, dtype=np.int32).reshape((-1,1,2))
            polygon = map_polygon_to_original(
                np.array(polygon),
                transform
            )
            polygon = polygon.reshape((-1,1,2))
            x, y, w, h = cv2.boundingRect(polygon)     
            cv2.rectangle(
                highlighted,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),          # Red
                thickness=2,
            )


            # --------------------------------------------------
            # Label
            # --------------------------------------------------
            label = "Damage"

            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            
            # Keep label inside image
            text_y = max(y - 10, th + 5)

            # Background for text
            cv2.rectangle(
                highlighted,
                (x, text_y - th - baseline - 4),
                (x + tw + 8, text_y + 2),
                (0, 0, 255),
                -1,
            )
            
            cv2.putText(
                highlighted,
                label,
                (x + 4, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


    # ----------------------------------------------------------
    # Display
    # ----------------------------------------------------------
    if show:
        plt.figure(figsize=(7, 7))
        plt.imshow(highlighted)
        plt.title("Detected Cuts / Damages")
        plt.axis("off")
        plt.show()

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    plt.imsave(save_path, highlighted, dpi=dpi,)

    print(f"✓ Saved to: {save_path}")
    print(f"✓ DPI: {dpi}")

    return highlighted


# ------------------------------------------------------------------
# Example
# ------------------------------------------------------------------
if __name__ == "__main__":
    highlighted = highlight_cuts(
        r"../generated-results/2024-12-24_10_10_40_(18130)_sardine_input_segmented_1.png", # replace with your image path
        save_path=r"../generated-results/sardine_fish_cuts.png",
        conf=0.85,
        dpi=1200,
    )
    
    
    highlighted = highlight_cuts(
        r"../generated-results/2025-01-28_10_50_33_(19200)_mackerel_input_segmented_0.png", # replace with your image path
        save_path=r"../generated-results/mackerel_fish_cuts.png",
        conf=0.85,
        dpi=1200,
    )
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model
import base64
from ultralytics import YOLO
from . import Species_Classification, sardine, mackerel, prawns, size
import logging
from logging.handlers import RotatingFileHandler


# Get the logger instance
logger = logging.getLogger('updated_outputs')
logger.setLevel(logging.INFO)

# Create a RotatingFileHandler
rotating_handler = RotatingFileHandler(
    filename='updated_output.log',  # Log file name for the rotating handler
    mode='a',
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,  # Keep 3 backup files
    encoding=None,
    delay=0
)

# Define a log format for the rotating handler
Log_Format = "%(asctime)s %(levelname)s %(message)s"
formatter = logging.Formatter(Log_Format)
rotating_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(rotating_handler)

try:
    yolo = YOLO(r'fish_product_models/best.pt')
    prawn = YOLO(r'fish_product_models/prawn.pt')
    cut = YOLO(r'fish_product_models/cut-seg.pt')
    scale_model = YOLO(r'fish_product_models/scale_detection_model.pt')
    logger.info("YOLO models loaded successfully from local path.")
except Exception as e:
    yolo = YOLO(r'/home/ec2-user/fish_product_models/best.pt')
    prawn = YOLO(r'/home/ec2-user/fish_product_models/prawn.pt')
    cut = YOLO(r'/home/ec2-user/fish_product_models/cut-seg.pt')
    scale_model = YOLO(r'/home/ec2-user/fish_product_models/scale_detection_model.pt')
    logger.info("YOLO models loaded successfully from EC2 path.")
    logger.error(f"Error loading YOLO models from local path: {e}")

def extract_single_image_segment2(img, mask_segment, shape):
    logger.info("Extracting image segments for individual Fish/Prawn. -->V2")
    height, width = shape[0], shape[1]
    segment = np.array(mask_segment, dtype=np.int32)
    segment = segment.reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(segment)
    segment_image = np.zeros((h, w, 3), dtype=np.uint8)
    segment_image[0:h, 0:w] = masked_img[y:y+h, x:x+w]
    black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
    segment_image[black_mask] = [255, 255, 255]
    logger.info("Extracted segment with white background.")
    return segment_image

def extract_single_image_segment(img, mask_segment, shape, background='black'):
    logger.info("Extracting image segments for individual Fish/Prawn.")
    height, width = shape[0], shape[1]
    segment = np.array(mask_segment, dtype=np.int32)
    segment = segment.reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(segment)
    segment_image = np.zeros((h, w, 3), dtype=np.uint8)
    segment_image[0:h, 0:w] = masked_img[y:y+h, x:x+w]

    if background.lower() == 'black':
        black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
        segment_image[black_mask] = [0, 0, 0]  # Set background to black
        logger.info("Extracted segment with black background.")
    elif background.lower() == 'white':
        black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
        segment_image[black_mask] = [255, 255, 255]  # Set background to white
        logger.info("Extracted segment with white background.")
    else:
        logger.error("Invalid background color specified.")
        raise ValueError("Invalid background color. Use 'black' or 'white'.")

    return segment_image

def process_image2(image, size=(640, 640)): # for eye model
    logger.info("Pre-processing image for damage detection model.")
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = size[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = size[1]
        new_w = int(new_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))
    logger.debug(f"Image resized to {new_w}x{new_h}.")

    pad_h = (size[1] - new_h) // 2
    pad_w = (size[0] - new_w) // 2
    padded_image = cv2.copyMakeBorder(
        resized_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    padded_image = cv2.resize(padded_image, (640,640))
    logger.debug("Image padded and resized to 640x640.")
    return np.asarray(padded_image)

def is_cut(image):
    logger.info("Running Damage detection model.")
    img = image[:,:,::-1]
    res = cut.predict(img, conf=0.5)
    if len(res[0].boxes.xyxy) == 0:
        logger.info("No Damage Detected.")
        return False
    else:
        logger.info("Fish is Damaged.")
        return True

def return_mask(image, masks, labels, boxes):
    logger.info("Applying masks and labels to the predicted image.")

    # Ensure the image is writable by creating a copy
    writable_image = np.copy(image)

    mask_image = np.zeros_like(writable_image)

    height, width, _ = writable_image.shape

    for mask, label, box in zip(masks, labels, boxes):
        xmin, ymin, xmax, ymax = list(map(int, box))

        # Calculate the thickness and text size based on image dimensions
        thickness = max(int(min(height, width) / 200), 1)
        text_size = max(int(min(height, width) / 800), 1)

        if 'good' in label.lower():
            color = (0, 255, 0)  # Green mask
        elif 'ok' in label.lower():
            color = (255, 255, 0)  # Yellow mask
        else:
            color = (255, 0, 0)  # Red mask

        mask_coords = np.array(mask, dtype=np.int32)
        cv2.fillPoly(mask_image, [mask_coords], color)
        cv2.rectangle(writable_image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(writable_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness)

    alpha = 0.2  # Transparency level for the mask
    image_with_masks = cv2.addWeighted(writable_image, 1 - alpha, mask_image, alpha, 0)
    image_with_masks = cv2.cvtColor(image_with_masks, cv2.COLOR_BGR2RGB)
    logger.info("Masks and labels applied to the output image.")
    return image_with_masks


def freshness_prediction(image, w_mask_image, b_mask_image, fish, box):
    """
    Predict the freshness of a fish based on its appearance and characteristics.

    Args:
    - image: Original image containing the fish.
    - w_mask_image: Image segment with white background.
    - b_mask_image: Image segment with black background.
    - fish: Species of the fish.
    - box: Bounding box coordinates of the fish.

    Returns:
    - label: Freshness label ('Good' or 'Bad').
    - reason: Reason for the freshness label, if any.
    """

    logger.info(f"Starting freshness prediction for {fish}.")

    # Check the species of the fish and perform freshness prediction accordingly
    if fish.lower() == 'sardine':
        try:
            # Check for size in sardines
            scale_size, reason = scale_model.predict(w_mask_image)
            if scale_size == 'Bad':
                logger.info("Sardine has small size.")
                return 'Bad', 'Size'
            else:
                logger.info("Not small size!! Proceeding with sardine-softness prediction.")
                pred = sardine.predict_sardine(b_mask_image)
                logger.info(f"Sardine freshness prediction: {pred}.")
                if pred == 'Bad':
                    return pred, 'Softness'
                else:
                    return pred, None
        except Exception as e:
            logger.error(f"Error during scale detection for sardine: {e}")
            # Predict freshness based on sardine-specific model
            pred = sardine.predict_sardine(b_mask_image)
            logger.info(f"Sardine freshness prediction (exception handling): {pred}.")
            if pred == 'Bad':
                return pred, 'Softness'
            else:
                return pred, None

    elif fish.lower() == 'mackerel':
        # Check if the fish is damaged
        damage = is_cut(process_image2(w_mask_image))
        if damage:
            logger.info("Mackerel is damaged.")
            return 'Bad', 'Damaged'
        else:
            # Predict freshness based on mackerel-specific model
            pred = mackerel.predict_mackerel(b_mask_image)
            logger.info(f"Mackerel freshness prediction: {pred}.")
            if pred == 'Bad':
                return pred, 'Softness'
            else:
                return pred, None

    elif fish.lower().replace(" ", "") == 'whiteprawn':
        # Predict freshness based on white prawn-specific model
        pred = prawns.predict_prawn(b_mask_image)
        logger.info(f"White prawn freshness prediction: {pred}.")
        if pred == 'Bad':
            return pred, 'Softness'
        else:
            return pred, None

    else:
        # Default prediction for other fish species
        damage = is_cut(process_image2(w_mask_image))
        if damage:
            logger.info("Fish is damaged.")
            return 'Bad', 'Damaged'
        else:
            logger.info("Fish is in good condition (default prediction).")
            return 'Good', None

def final_prediction(image, species, model):
    """
    Perform final prediction on the given image for a specific species using a specified model.

    Args:
    - image: Image file path or image data (numpy array).
    - species: Species label for classification.
    - model: Model name ('fish' or 'prawn').

    Returns:
    - result: A dictionary containing the prediction results.
    """

    logger.info(f"Starting freshness prediction for species: {species} using model: {model}.")

    result = dict()
    top3 = []

    # Open image and convert it to numpy array
    try:
        image = Image.open(image)
        image = np.asarray(image)
        logger.info("Image successfully opened and converted to numpy array.")
    except Exception as e:
        logger.error(f"Error opening or converting image: {e}")
        return None

    shape = image.shape

    try:
        # Perform object detection based on model
        if model.lower() == 'fish':
            pred = yolo.predict(image[:, :, ::-1], conf=0.75)
        else:
            pred = prawn.predict(image[:, :, ::-1], conf=0.5)
        logger.info("Fish/Prawn detection completed.")
    except Exception as e:
        logger.error(f"Error during object detection: {e}")
        return None

    # Return None if no boxes or masks are detected
    if len(pred[0].boxes.xyxy) == 0 or pred[0].masks == None:
        logger.warning("No boxes or masks detected in the image.")
        return None

    # Extract bounding boxes and masks
    boxes = pred[0].boxes.xyxy
    masks = pred[0].masks.xy
    filtered_boxes = sorted(boxes, key=lambda bbox: bbox[0])
    sorted_masks = [masks[i] for i in sorted(range(len(masks)), key=lambda x: boxes[x][0])]

    new_labels = []
    reasons = []
    i = 0

    # Iterate over each detected box and mask
    for box, mask in zip(filtered_boxes, sorted_masks):
        i += 1
        xmin, ymin, xmax, ymax = list(map(int, box))
        extracted_image = image[ymin:ymax, xmin:xmax]

        # Perform freshness prediction
        w_mask_image = extract_single_image_segment(image, mask, shape=shape, background='white')
        b_mask_image = extract_single_image_segment(image, mask, shape=shape, background='black')
        L, R = freshness_prediction(image, w_mask_image, b_mask_image, species, box)

        # Construct label for the prediction
        v = str(i) + '-' + str(L)
        if R is not None:
            v += '-' + str(R)
        reasons.append(R)
        new_labels.append(v)
        logger.info(f"Prediction for fish {i}: {v}")

    # Generate output image with labels
    output_image = return_mask(image, sorted_masks, new_labels, filtered_boxes)
    _, image_bytes = cv2.imencode('.jpg', output_image)
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')

    # Count good and bad predictions
    good = 0
    bad = 0
    for i in range(len(new_labels)):
        fresh = new_labels[i].split('-')[1]
        top3.append(fresh)
        if fresh.lower() == 'good':
            good += 1
        else:
            bad += 1

    # Construct result dictionary
    result['fishes-detected'] = good + bad
    result['good-fishes'] = good
    result['bad-fishes'] = bad
    result['reasons'] = reasons
    result['image-encode'] = encoded_string
    result['classification'] = 'True'
    result['species'] = species
    result['first3'] = top3

    logger.info(f"Final prediction results: Fishes detected:{good+bad} for species {species}, Good fishes:{good} and Bad fishes: {bad}, Reasons: {reasons}")

    return result

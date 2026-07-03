from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2, os, math, pickle
import uuid
from PIL import Image

try:
    scale_model = YOLO(r'fish_product_models/scale_detection_model.pt')
    fish_model = YOLO(r'fish_product_models/best.pt')
    model_path = r'fish_product_models/size.pkl'
except:
    scale_model = YOLO(r'/home/ec2-user/fish_product_models/scale_detection_model.pt')
    fish_model = YOLO(r'/home/ec2-user/fish_product_models/best.pt')
    model_path = r'/home/ec2-user/fish_product_models/size.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

def save_image(image_input, folder_path=r'temp/'):
    os.makedirs(folder_path, exist_ok=True)
    unique_filename = str(uuid.uuid4()) + ".jpeg"
    image_path = os.path.join(folder_path, unique_filename)
    image = Image.fromarray(image_input)
    image.save(image_path, "JPEG")
    return image_path

def get_fish_height(box):
  x1, y1, x2, y2 = map(float, box)
  h = abs(y2 - y1)
  w = abs(x2 - x1)
  if h<w: (h, w) = (w, h)
  d = math.sqrt(w**2 + h**2)
  return h, w, d


def circumference_area(img):
    # Load the segmented fish image
    # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure at least one contour was found
    if len(contours) > 0:
        # Get the largest contour (presumably the fish)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the circumference of the contour
        circumference = cv2.arcLength(largest_contour, closed=True)

        # Calculate the area enclosed by the contour
        area = cv2.contourArea(largest_contour)

        return circumference, area
    
# def get_segmented_img(pred, index=0):
#     img_shape = pred.orig_shape
#     img = pred.orig_img.copy()
#     height, width = img_shape[0], img_shape[1]
#     masks = pred.masks.xy
#     seg_img_list = []
#     for i in range(len(masks)):
#         if i != index: continue
#         mask = masks[i]
#         mask_points = mask.astype(int)
#         binary_mask = np.zeros((height, width), dtype=np.uint8)
#         cv2.fillPoly(binary_mask, [mask_points], 255)
#         masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
#         x, y, w, h = cv2.boundingRect(mask_points)
#         segment_image = np.zeros((height, width, 3), dtype=np.uint8)
#         segment_image[y:y+h, x:x+w] = masked_img[y:y+h, x:x+w]
#         segment_image = segment_image[y:y+h, x:x+w]
#         white_background = False
#         if white_background:
#             black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
#             segment_image[black_mask] = [255, 255, 255]
#         seg_img_list.append(segment_image)
#     return seg_img_list


# def get_object_height(img_path, model, conf=0.5, iou=0.7, seg=False):
#     seg_img_list = None
#     results = model(img_path, conf=conf, iou=iou, verbose=False)
#     if len(results[0].boxes.cls) != 0:
#         index = 0
#         hp = int(results[0].boxes.xywh[index][3])
#         wp = int(results[0].boxes.xywh[index][2])
#         dp = int(math.hypot(hp, wp))
#         if hp<wp: (hp, wp) = (wp, hp)
#         obj_present = True
#         if seg: seg_img_list = get_segmented_img(results[0], index)
#     else:
#         hp, wp, dp = (0, 0, 0)
#         obj_present = False
#     return hp, wp, dp, obj_present, seg_img_list


def fish_measures_v2(sh, sw, sd, box):
    # Predict with the scale model
    scale_hp, scale_wp, scale_dp, scale = sh, sw, sd, True
    # Predict with the fish model
    fish_hp, fish_wp, fish_dp, fish, fish_img_list = *get_fish_height(box), True, None

    try:
        scale_hcm = 13
        one_pixel_value = scale_hcm / scale_hp
        fish_hcm = int(fish_hp * one_pixel_value)
        fish_wcm = int(fish_wp * one_pixel_value)
        fish_dcm = int(fish_dp * one_pixel_value)
        scale_dcm = int(scale_dp * one_pixel_value)
        scale_wcm = int(scale_wp * one_pixel_value)
    except:
        (fish_hcm, fish_wcm, fish_dcm,
         scale_dcm, scale_wcm, one_pixel_value) = (0, 0, 0, 0, 0, 0)

    return (fish_hcm, fish_wcm, fish_dcm, scale_dcm, scale_wcm,
            fish, scale, fish_img_list, one_pixel_value)


def single_img_data_v2(img_path, segment, sh, sw, sd, box):
    filename = os.path.basename(img_path)
    (fish_hcm, fish_wcm, fish_dcm, scale_dcm, scale_wcm, fish, scale,
     fish_img_list, one_pixel_value) = fish_measures_v2(sh, sw, sd, box)
    data = []
    circumference, area = circumference_area(img = segment)
    if one_pixel_value != 0:
      circumference = int(circumference * one_pixel_value)
      area = int(area * one_pixel_value * one_pixel_value)
    data.append((filename, fish_hcm, fish_wcm, fish_dcm,
                scale_dcm, scale_wcm, one_pixel_value,
                scale, fish, circumference, area))
    return data

def pred_single_img(test, model):
    probabilities = []
    for index, row in test.iterrows():
        if row['Scale']==False:
            print(f'Scale not detected @ {index}')
            break # go to softness model
        if row['Fish']==False:
            print(f'Fish not detected @ {index}')
            break
        Height = row['FishHeight']
        Width = row['FishWidth']
        diagonal = row['FishDiagonal']
        Circumference = row['Circumference']
        Area = row['Area']
        X_test = [[Height, Width, diagonal, Circumference, Area]]
        # pred = model.predict(X_test)
        # Get probability estimates for the predicted labels
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            # print(f"Probability :", y_proba)
        probabilities.append(y_proba)
    return probabilities

def get_object_height_v2(img_path, model, conf=0.5, iou=0.7, seg=False):
    seg_img_list = None
    results = model(img_path, conf=conf, iou=iou, verbose=False)
    if len(results[0].boxes.cls) != 0:
        index = 0
        hp = int(results[0].boxes.xywh[index][3])
        wp = int(results[0].boxes.xywh[index][2])
        dp = int(math.hypot(hp, wp))
        if hp<wp: (hp, wp) = (wp, hp)
        obj_present = True
    else:
        hp, wp, dp = (0, 0, 0)
        obj_present = False
    return hp, wp, dp, obj_present


def size_prediction(img_array,segment,box):
    try:
        print('triggered!!!!!!!!')  
        img_path = save_image(img_array)
        sh, sw, sd, scale = get_object_height_v2(img_path, scale_model)
        data = single_img_data_v2(img_path, segment, sh, sw, sd, box)
        df = pd.DataFrame(data, columns=['Filename', 'FishHeight', 'FishWidth',
                                        'FishDiagonal', 'ScaleDiagonal', 'ScaleWidth',
                                        'OnePixelValue', 'Scale', 'Fish',
                                        'Circumference', 'Area'])
        probabilities = pred_single_img(df, model)
        threshold = 0.8
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                print(e)
        # Adjust predictions based on the threshold
        predictions = np.ones(len(probabilities))
        for i in range(len(probabilities)):
            if probabilities[i][0][0] >= threshold:
                predictions[i] = 0
        label_map = {0: 'Bad', 1: 'Good'}
        for pred in predictions :
            label = label_map[int(pred)]

        if label == 'Good':
            print(f'size->{label}')
            return 'Good', None
        else:
            print(f'size->{label}')
            return 'Bad', 'Size'
    except Exception as e:
        print(e)
        return 'Good', None
      

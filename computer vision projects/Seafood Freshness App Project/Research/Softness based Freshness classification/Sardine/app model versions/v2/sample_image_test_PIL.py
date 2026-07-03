# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:16:33 2023

@author: sowmya
"""

import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from PIL import Image


def calc_cm_vals(y_true, y_pred, class_index=None):
    pred = tf.argmax(y_pred, axis=1)
    true = tf.reshape(y_true, (-1,))
    
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, class_index), 
                                              tf.equal(pred, class_index)), 
                                               tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(true, class_index), 
                                              tf.equal(pred, class_index)), 
                                               tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, class_index), 
                                             tf.not_equal(pred, class_index)), 
                                              tf.float32))
    return tp, fp, fn

def recall(y_true, y_pred, class_index=1):
    tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
    return tp / (tp + fn + K.epsilon()) 

def precision(y_true, y_pred, class_index=0):
    tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
    return tp / (tp + fp + K.epsilon())  
    





def preprocess_img(image, target_size):
    if image is not None:
        height, width, channels = image.shape
        if width<height :
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)                
    preprocessed_image = cv2.resize(image, target_size)
    reshaped_image = preprocessed_image.reshape(1, *preprocessed_image.shape)

    image = reshaped_image.astype(np.float32)
    image = image / 255.0
    return image

def predict_image(preprocessed_img, model, label_map):
    prediction = model.predict(preprocessed_img)
    prediction = np.array([np.argmax(pred) for pred in prediction])  
    reverse_label_map = {idx: img_type for img_type, idx in label_map.items()}
    predicted_class = reverse_label_map[prediction[0]] 
    return predicted_class





def main(img_path = ''):
    model_path = r"D:\New folder\Fish-Data-Science-project-Jupyter-notebooks\New Experiments\fish_freshness_classification - Copy\saved_models\Sardine\freshness_saved_models\2011_1_densenet_single_dropout_0.2_recall_1_precision_0_accuracy_mobilenetv2_model.h5"
    label_map = {'Bad': 0, 'Good': 1}
    target_size = (280, 180)
    
    custom_metrics = {'recall_1':recall, 'precision_0':precision}
    model = load_model(model_path, custom_objects=custom_metrics)
    
    # Open an image file
    img = Image.open(img_path) 
    img = np.array(img)
    preprocessed_img = preprocess_img(img, target_size)
    predicted_class = predict_image(preprocessed_img, model, label_map)
    print(predicted_class)
    return predicted_class
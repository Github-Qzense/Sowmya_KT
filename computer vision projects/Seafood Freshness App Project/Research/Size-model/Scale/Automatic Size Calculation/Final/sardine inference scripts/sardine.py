# import pickle
import cv2
# import numpy as np
from keras.models import load_model
import tensorflow as tf
# import keras.backend as K
# import logging

try:
    model = load_model(r"fish_product_models/sardine_june.h5")
except:
    model = load_model(r"/home/ec2-user/fish_product_models/sardine_june.h5")
    
target_size = (224, 224)
threshold = 0.5 #0.478963

def process_image(image):
    resized_image = cv2.resize(image,target_size)
    processed_image = tf.keras.applications.densenet.preprocess_input(resized_image)
    processed_image = tf.expand_dims(processed_image, axis=0)
    return processed_image

def predict_sardine(image):
    image = process_image(image)
    pred = model.predict(image, verbose=0)
    print(pred[0][0])
    label = 'Good' if pred[0][0] >= threshold else 'Bad'
    return label

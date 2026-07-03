from django.shortcuts import render

# Create your views here.
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Image
from keras.models import load_model
import tensorflow as tf
from .serializers import ImageSerializer
import cv2
from PIL import Image
import numpy as np

# Load the saved models
cnn_model = load_model('m96.h5') 

# Define the list of fish species (folder names)
fish_species = ['Mackerel', 'Seer', 'Sardine', 'Tuna']

def preprocess(img):
    # Convert Django ImageField object to OpenCV-compatible format
    pil_image = Image.open(img)
    img = np.array(pil_image)
    print(f"shape : {img.shape}")
            
    image_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    target_size = (224, 224)
    image = cv2.resize(image_cv, target_size)  # Resize the image to match the input size of the CNN model
    image = image / 255.0  # Normalize the image

    # Expand dimensions to match the input shape of the CNN model
    image = np.expand_dims(image, axis=0)    
    return image


class ImageClassificationView(APIView):
    def post(self, request, format=None):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            print(f"Uploaded file name: {image.name}")            
            img_preprocessed = preprocess(image)
            prediction = np.argmax(cnn_model.predict(img_preprocessed))
            prediction = fish_species[prediction]

            # Return the classification result as a response
            return Response({'result': prediction}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

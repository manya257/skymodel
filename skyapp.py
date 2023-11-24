
# Import libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image, ImageOps
import numpy as np

# Set Streamlit options
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the pre-trained Saturn-Jupiter-Moon classification model
@st.cache(allow_output_mutation=True)
def load_model():
    model = load_model('/content/drive/MyDrive/saturn_jupiter_moon_classifier.h5')  # Adjust the path to your model
    return model

model = load_model()

# Streamlit app title and description
st.write("# Saturn-Jupiter-Moon Image Classifier")
st.write("Upload an image, and the classifier will predict whether it's Saturn, Jupiter, or Moon.")

# File uploader
file = st.file_uploader("Please upload an image", type=["jpg", "png"])

# Function to make predictions
def predict_class(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = ImageOps.exif_transpose(np.array(img))
    img_array = img_array / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)

    return prediction

# Display results
if file is not None:
    st.image(file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = predict_class(file)

    # Display predicted class
    predicted_class = np.argmax(prediction)
    
    if predicted_class == 0:
        st.write("**Prediction:** Moon")
    elif predicted_class == 1:
        st.write("**Prediction:** Jupiter")
    elif predicted_class == 2:
        st.write("**Prediction:** Saturn")

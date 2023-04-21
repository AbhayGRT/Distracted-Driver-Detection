import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('weights_best.h5')

# Map the class labels to their respective meanings
labels = {
    '0': 'safe driving',
    '1': 'texting - right',
    '2': 'talking on the phone - right',
    '3': 'texting - left',
    '4': 'talking on the phone - left',
    '5':'operating the radio',
    '6': 'drinking',
    '7': 'reaching behind',
    '8': 'hair and makeup',
    '9': 'talking to passenger'
}

# Define a function to preprocess the input image
def preprocess_image(image_path):
    # Load the image in grayscale and resize it to match the input size of the model
    image = load_img(path=image_path, color_mode="grayscale", target_size=(160, 120))
    # Convert the image to a numpy array
    input_arr = img_to_array(image)
    # Add an extra dimension to the array to represent the batch size of 1
    input_arr = np.array([input_arr])
    # Preprocess the image by rescaling the pixel values
    input_arr = input_arr / 255.0
    return input_arr

# Define the Streamlit app
def app():
    # Set the title and description of the app
    st.title("Driver Distraction Detection")
    st.write("This app can detect driver distractions in images using a deep learning model.")
    
    # Add a file uploader to allow the user to select an image
    st.write("Please select an image to upload:")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    # If the user has uploaded an image
    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image and make a prediction
        input_data = preprocess_image(uploaded_file)
        prediction = model.predict(input_data)[0]
        predicted_index = np.argmax(prediction)
        if str(predicted_index) in labels:
            predicted_class = labels[str(predicted_index)]
        else:
            predicted_class = f"Class {predicted_index}"

        
        # Display the predicted class and confidence score
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence score: {prediction.max()*100:.2f}%")

# Run the app
app()

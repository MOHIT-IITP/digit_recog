import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mnist_cnn_model.h5")

# Streamlit app
st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a handwritten digit (0–9)")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert: black background with white digit
    image = image.resize((28, 28))  # Resize to 28x28
    st.image(image, caption='Processed Image', width=150)

    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    if st.button("Predict"):
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        st.success(f"Predicted Digit: {predicted_label}")


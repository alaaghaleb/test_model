import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your TensorFlow model
model = tf.keras.models.load_model("inception.model.h5")

# Map prediction to disease names
disease_names = {
    0: "Bacterial Spot",
    1: "Early Blight",
    2: "Late Blight",
    3: "Leaf Mold",
    4: "Septoria Leaf Spot",
    5: "Spider Mites",
    6: "Target Spot",
    7: "Yellow Leaf Curl Virus",
    8: "Mosaic Virus",
    9: "Healthy"
}


def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image if needed
    x = np.array(image)
    x = x / 255.0  # Normalize pixel values
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x


def predict(image):
    try:
        # Make prediction using the loaded model
        preds = model.predict(image)
        pred_class = np.argmax(preds, axis=1)[0]

        predicted_disease = disease_names.get(pred_class, "Unknown")
        return predicted_disease
    except Exception as e:
        return str(e)


st.title("Disease Predictor")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image file
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(pil_image)

    # Make prediction
    predicted_disease = predict(processed_image)

    st.success(f"Prediction: {predicted_disease}")

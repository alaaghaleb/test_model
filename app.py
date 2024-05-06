from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model("inception.model.h5")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file was uploaded
        if 'image_file' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        # Get the uploaded image file
        image_file = request.files['image_file']

        # Ensure the file is an image
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Unsupported file format. Only PNG and JPEG images are supported.'}), 400

        # Read the image file
        image_data = image_file.read()

        # Convert image data to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        pil_image = pil_image.resize((224, 224))  # Resize image if needed
        x = np.array(pil_image)
        x = x / 255.0  # Normalize pixel values
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        # Make prediction using the loaded model
        preds = model.predict(x)
        pred_class = np.argmax(preds, axis=1)[0]
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
            9: "healthy"
        }
        predicted_disease = disease_names.get(pred_class, "Unknown")
        if pred_class not in [0,1,2,3,4,5,6,7,8,9]:
            return jsonify({'wrongprediction': "Please enter a good iamge"}), 200
        else:
            return jsonify({'prediction': predicted_disease}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run Flask app on a publicly accessible server

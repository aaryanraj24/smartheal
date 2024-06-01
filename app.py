from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('segmentation_model (2).h5')  # Adjust the path

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        # You would add your image processing and prediction code here
        # For simplicity, let's assume it returns a JSON with coordinates
        coordinates = {'x': 100, 'y': 150, 'width': 200, 'height': 100}
        return jsonify(coordinates)

if __name__ == '__main__':
    app.run(debug=True)

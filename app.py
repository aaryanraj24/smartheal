from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

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
        
        # Predict the segmented image
        segmented_image, _ = predict_image(file_path)

        # Save the segmented image to a byte stream
        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Correct keyword for Flask version 2.0 and above
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='segmented.png')

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image = image.resize((224, 224))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        prediction = model.predict(image_array)
        mask = prediction[0, :, :, 0] if prediction.shape[-1] == 1 else prediction[0]

        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
        mask = np.array(mask_image) / 255.0

        seg_image = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
        seg_image[mask > 0.5] = [0, 255, 0]  # Adjust the threshold and colors as needed

        combined = Image.blend(Image.open(image_path), Image.fromarray(seg_image), alpha=0.5)
        return combined, mask
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

if __name__ == '__main__':
    app.run(debug=True)

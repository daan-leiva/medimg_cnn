from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import io
from models.cnn_model import MedCNN
from models.cnn_model_large import MedCNN_Large
import base64
from io import BytesIO
from visualizations.gradcam import GradCAM, overlay_heatmap
import time

app = Flask(__name__)

# -------------------- Model Loading --------------------
# Load small CNN model
SMALL_MODEL_PATH = "results/small_cnn/medcnn_chestxray_v2.pt"
small_model = MedCNN(num_classes=2)
small_model.load_state_dict(torch.load(SMALL_MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
small_model.eval()  # set to eval mode

# Load large CNN model
LARGE_MODEL_PATH = "results/large_cnn/medcnn_chestxray.pt"
large_model = MedCNN_Large(num_classes=2)
large_model.load_state_dict(torch.load(LARGE_MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
large_model.eval()  # set to eval mode


# -------------------- Image Preprocessing --------------------
def preprocess_image(file):
    """
    Converts input image to grayscale, resizes, normalizes, and returns a tensor.
    """
    try:
        image = Image.open(file).convert('L')  # convert to grayscale
    except Exception as e:
        print(e)
        raise ValueError("Failed in preprocess img")
    
    # Apply resizing, tensor conversion, and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # input size for the CNN
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # match training distribution
    ])
    
    return transform(image).unsqueeze(0)  # add batch dimension


# -------------------- Prediction Endpoint --------------------
@app.route('/predict_medimg', methods=['POST'])
def predict_medimg():
    # Get image and model size from the request
    image_file = request.files.get('image')
    model_size = request.form.get('model_size', '').lower()

    # Basic request validation
    if image_file is None:
        return jsonify({'error': 'No image provided'}), 400
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if model_size not in ['small', 'large']:
        return jsonify({'error': 'Invalid model size'}), 400

    try:
        # Preprocess the input image
        input_tensor = preprocess_image(image_file)

        # Select the correct model
        model = small_model if model_size == 'small' else large_model

        # -------------------- Inference --------------------
        start_time = time.time()  # start timing

        with torch.no_grad():
            output = model(input_tensor)  # forward pass
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        inference_time = time.time() - start_time  # elapsed time in seconds

        # -------------------- Grad-CAM Heatmap --------------------
        cam = GradCAM(model, model.features[-3])  # use appropriate convolutional layer
        heatmap = cam.generate(input_tensor, class_idx=predicted_class)  # class-specific heatmap
        overlay = overlay_heatmap(input_tensor, heatmap)  # overlay heatmap on image

        # Convert heatmap overlay to base64
        buf = BytesIO()
        Image.fromarray(overlay).save(buf, format='PNG')
        base64_heatmap = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Return structured JSON response
        return jsonify({
            'prediction': int(predicted_class),
            'confidence': round(confidence * 100, 2),  # percentage
            'heatmap': base64_heatmap,  # base64 encoded PNG
            'inference_time': round(inference_time * 1000, 2)  # milliseconds
        })

    except Exception as e:
        # Catch all errors and return as JSON
        return jsonify({'error': str(e)}), 500


# -------------------- Run Locally --------------------
if __name__ == '__main__':
    # Run the Flask server for local development
    app.run(host='0.0.0.0', port=5000, debug=True)
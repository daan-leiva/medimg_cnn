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

# Load model
SMALL_MODEL_PATH = "results/small_cnn/medcnn_chestxray_v2.pt"
small_model = MedCNN(num_classes=2)
small_model.load_state_dict(torch.load(SMALL_MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
small_model.eval()
LARGE_MODEL_PATH = "results/large_cnn/medcnn_chestxray.pt"
large_model = MedCNN_Large(num_classes=2)
large_model.load_state_dict(torch.load(LARGE_MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
large_model.eval()

# Preprocessing function
def preprocess_image(file):
    try:
        image = Image.open(file).convert('L')
    except Exception as e:
        print(e)
        raise ValueError("Failed in preprocess img")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # or your model’s input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # match training
    ])
    return transform(image).unsqueeze(0)

@app.route('/predict_medimg', methods=['POST'])
def predict_medimg():
    image_file = request.files.get('image')
    model_size = request.form.get('model_size', '').lower()
    if image_file is None:
        return jsonify({'error': 'No image provided'}), 400
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if model_size not in ['small', 'large']:
        return jsonify({'error': 'Invalid model size'}), 400

    try:
        input_tensor = preprocess_image(image_file)
        if model_size == 'small':
            model = small_model
        else:
            model = large_model

        start_time = time.time() # to calculate inference time
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        inference_time = time.time() - start_time  # ← inference timing

        # Grad-CAM
        cam = GradCAM(model, model.features[-3])  # adjust layer if needed
        heatmap = cam.generate(input_tensor, class_idx=predicted_class)
        overlay = overlay_heatmap(input_tensor, heatmap)

        # Encode overlay image to base64
        buf = BytesIO()
        Image.fromarray(overlay).save(buf, format='PNG')
        base64_heatmap = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': int(predicted_class),
            'confidence': round(confidence * 100, 2),
            'heatmap': base64_heatmap,
            'inference_time': round(inference_time * 1000, 2)  # milliseconds
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import torch
import argparse
from models.cnn_model import MedCNN
from models.cnn_model_large import MedCNN_Large
import os

# === Constants ===
SMALL_MODEL_PATH = 'results/small_cnn/medcnn_chestxray_v2.pt'
LARGE_MODEL_PATH = 'results/large_cnn/medcnn_chestxray.pt'
EXPORT_DIR = 'exports'

def export_to_onnx(model_size):
    # Ensure exports directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Choose model and path
    if model_size == 'small':
        model = MedCNN(num_classes=2)
        model_path = SMALL_MODEL_PATH
        export_path = f'{EXPORT_DIR}/medcnn_small.onnx'
    elif model_size == 'large':
        model = MedCNN_Large(num_classes=2)
        model_path = LARGE_MODEL_PATH
        export_path = f'{EXPORT_DIR}/medcnn_large.onnx'
    else:
        raise ValueError("model_size must be 'small' or 'large'")

    # Load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 1, 224, 224)

    # Export
    torch.onnx.export(
        model, dummy_input, export_path,
        input_names=["input"], output_names=["output"],
        opset_version=11
    )

    print(f"Exported {model_size} model to {export_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export MedCNN model to ONNX")
    parser.add_argument('--model_size', type=str, required=True, choices=['small', 'large'],
                        help="Model size to export: 'small' or 'large'")
    args = parser.parse_args()

    export_to_onnx(args.model_size)
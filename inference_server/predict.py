import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

from models.cnn_model import MedCNN
from models.cnn_model_large import MedCNN_Large
from visualizations.gradcam import GradCAM, overlay_heatmap
from utils.logger import Logger

# === Configuration ===

IMG_SIZE = 224  # Input size expected by CNN
CLASS_NAMES = ['Normal', 'Pneumonia']  # Class labels
SMALL_MODEL_PATH = 'results/small_cnn/medcnn_chestxray_v2.pt'  # Path to small model weights
LARGE_MODEL_PATH = 'results/large_cnn/medcnn_chestxray.pt'     # Path to large model weights
DEFAULT_IMAGE = 'data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg'  # Fallback image path

# === Preprocessing pipeline ===

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale channel
    transforms.Resize((IMG_SIZE, IMG_SIZE)),       # Resize for model input
    transforms.ToTensor(),                         # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])    # Match training distribution
])

# === Load and preprocess image ===
def load_image(img_path, logger):
    """
    Loads and preprocesses the image from the given path.
    """
    logger.info(f"Loading image: {img_path}")
    if not os.path.exists(img_path):
        logger.info(f"Image not found: {img_path}")
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = Image.open(img_path).convert('L')  # Load image as grayscale
    tensor = transform(image).unsqueeze(0)     # Add batch dimension
    return tensor

# === Run model prediction ===
def predict(model, image_tensor, device, logger):
    """
    Runs prediction on the provided image tensor using the given model.
    """
    logger.info("Running prediction")
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)              # Move to device
        outputs = model(image_tensor)                       # Forward pass
        probs = torch.softmax(outputs, dim=1)               # Convert to probabilities
        class_idx = torch.argmax(probs, dim=1).item()       # Predicted class
        confidence = probs[0, class_idx].item()             # Confidence score
    return CLASS_NAMES[class_idx], confidence, class_idx

# === Main CLI entry point ===
def main(args):
    """
    Main entry point for CLI execution. Loads model, processes image, predicts class,
    and optionally visualizes GradCAM.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = Logger(log_file='predict.log')
    logger.info("=== Inference started ===")

    # Select model and load weights
    if args.model_size == 'small':
        model = MedCNN(num_classes=2)
        MODEL_PATH = SMALL_MODEL_PATH
    else:
        model = MedCNN_Large(num_classes=2)
        MODEL_PATH = LARGE_MODEL_PATH

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    logger.info(f"Model loaded from {MODEL_PATH}")

    # Load and preprocess image
    img_path = args.img if args.img else DEFAULT_IMAGE
    image_tensor = load_image(img_path, logger)

    # Run prediction
    label, confidence, class_idx = predict(model, image_tensor, device, logger)
    logger.info(f"Prediction: {label} ({confidence*100:.2f}% confidence)")

    # Optionally visualize and save/display GradCAM
    if args.overlay:
        image_tensor = image_tensor.to(device)
        cam = GradCAM(model, target_layer=model.features[-3])  # Select final conv layer
        heatmap = cam.generate(image_tensor, class_idx)        # Generate GradCAM heatmap
        overlay = overlay_heatmap(image_tensor, heatmap)       # Overlay heatmap on image

        if args.save:
            # Save to file
            Path("predict_outputs").mkdir(parents=True, exist_ok=True)
            input_name = os.path.basename(img_path)
            base_name = os.path.splitext(input_name)[0]
            save_path = f"predict_outputs/{base_name}_overlay.png"

            import matplotlib
            import matplotlib.pyplot as plt
            if not args.save:
                matplotlib.use('Agg')  # For non-GUI environments
            plt.imsave(save_path, overlay)
            logger.info(f"Saved GradCAM overlay to {save_path}")
        else:
            # Display interactively
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 5))
            plt.imshow(overlay)
            plt.axis('off')
            plt.title("Grad-CAM")
            plt.show()
            logger.info("Displayed GradCAM overlay")

    logger.info("=== Inference finished ===")

# === CLI Interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pneumonia from chest X-ray")
    parser.add_argument('--img', type=str, help='Path to input image')
    parser.add_argument('--overlay', action='store_true', help='Display or save GradCAM overlay')
    parser.add_argument('--save', action='store_true', help='Save overlay to file instead of displaying')
    parser.add_argument('--model_size', default='small', type=str, help='Model size: small or large')

    args = parser.parse_args()

    # Validate model size
    if args.model_size not in ('small', 'large'):
        raise ValueError("Model size can only be large or small")

    main(args)
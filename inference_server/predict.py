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
IMG_SIZE = 224
CLASS_NAMES = ['Normal', 'Pneumonia']
SMALL_MODEL_PATH = 'results/small_cnn/medcnn_chestxray_v2.pt'
LARGE_MODEL_PATH = 'results/large_cnn/medcnn_chestxray.pt'
DEFAULT_IMAGE = 'data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg'  # Replace with an existing image

# === Preprocessing pipeline ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_image(img_path, logger):
    """
    Loads and preprocesses the image from the given path.

    Args:
        img_path (str): Path to the input image file.
        logger (Logger): Logger instance for logging events.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape [1, 1, H, W].
    """
    logger.info(f"Loading image: {img_path}")
    if not os.path.exists(img_path):
        logger.info(f"Image not found: {img_path}")
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = Image.open(img_path).convert('L')
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict(model, image_tensor, device, logger):
    """
    Runs prediction on the provided image tensor using the given model.

    Args:
        model (torch.nn.Module): Trained CNN model for classification.
        image_tensor (torch.Tensor): Input image tensor.
        device (torch.device): Device to run inference on (CPU or CUDA).
        logger (Logger): Logger instance for logging events.

    Returns:
        tuple: (predicted label as str, confidence as float, class index as int)
    """
    logger.info("Running prediction")
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, class_idx].item()
    return CLASS_NAMES[class_idx], confidence, class_idx

def main(args):
    """
    Main entry point for CLI execution. Loads model, processes image, predicts class,
    and optionally visualizes GradCAM.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = Logger(log_file='predict.log')
    logger.info("=== Inference started ===")

    if args.model_size == 'small':
        model = MedCNN(num_classes=2)
        MODEL_PATH = SMALL_MODEL_PATH
    else:
        model = MedCNN_Large(num_classes=2)
        MODEL_PATH = LARGE_MODEL_PATH

    # Load model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    logger.info(f"Model loaded from {MODEL_PATH}")

    # Load image
    img_path = args.img if args.img else DEFAULT_IMAGE
    image_tensor = load_image(img_path, logger)

    # Predict
    label, confidence, class_idx = predict(model, image_tensor, device, logger)
    logger.info(f"Prediction: {label} ({confidence*100:.2f}% confidence)")

    # GradCAM visualization
    if args.overlay:
        image_tensor = image_tensor.to(device)
        cam = GradCAM(model, target_layer=model.features[-3])
        heatmap = cam.generate(image_tensor, class_idx)
        overlay = overlay_heatmap(image_tensor, heatmap)

        if args.save:
            Path("predict_outputs").mkdir(parents=True, exist_ok=True)
            input_name = os.path.basename(img_path)
            base_name = os.path.splitext(input_name)[0]
            save_path = f"predict_outputs/{base_name}_overlay.png"

            import matplotlib
            import matplotlib.pyplot as plt
            if not args.save:
                matplotlib.use('Agg')
            plt.imsave(save_path, overlay)
            logger.info(f"Saved GradCAM overlay to {save_path}")
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 5))
            plt.imshow(overlay)
            plt.axis('off')
            plt.title("Grad-CAM")
            plt.show()
            logger.info("Displayed GradCAM overlay")

    logger.info("=== Inference finished ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pneumonia from chest X-ray")
    parser.add_argument('--img', type=str, help='Path to input image')
    parser.add_argument('--overlay', action='store_true', help='Display or save GradCAM overlay')
    parser.add_argument('--save', action='store_true', help='Save overlay to file instead of displaying')
    parser.add_argument('--model_size', default='small', type=str, help='Model size: small or large')

    args = parser.parse_args()

    if args.model_size not in ('small', 'large'):
        raise ValueError("Model size can only be large or small")
    main(args)

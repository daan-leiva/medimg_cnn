import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM object with model and layer of interest.

        Args:
            model (torch.nn.Module): The trained CNN model.
            target_layer (torch.nn.Module): The convolutional layer to extract activations and gradients from.
        """
        self.model = model
        self.model.eval()  # Ensure the model is in evaluation mode
        self.target_layer = target_layer

        self.gradients = None  # To store the gradients of the target layer
        self.activations = None  # To store the forward activations of the target layer

        # Register hooks to capture activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """
        Hook function to save the forward activation of the target layer.
        """
        self.activations = output.detach()  # Detach to avoid interfering with autograd

    def save_gradient(self, module, grad_input, grad_output):
        """
        Hook function to save the gradient of the target layer's output during backpropagation.
        """
        self.gradients = grad_output[0].detach()  # Only interested in gradient w.r.t. the output

    def generate(self, input_tensor, class_idx=None):
        """
        Generates the Grad-CAM heatmap for a given input image and class index.

        Args:
            input_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].
            class_idx (int, optional): Target class index. If None, uses predicted class.

        Returns:
            heatmap (np.ndarray): Heatmap of shape [H, W] with values in [0, 1].
        """
        output = self.model(input_tensor)  # Forward pass

        if class_idx is None:
            class_idx = torch.argmax(output)  # Default to highest scoring class

        self.model.zero_grad()  # Zero out any existing gradients
        output[:, class_idx].backward()  # Backpropagate for the selected class

        # Compute global average of gradients across spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Get the activations for the input (assume batch size = 1)
        activations = self.activations[0]

        # Weight each channel in the activations by the corresponding gradient
        activations = activations * pooled_gradients.view(-1, 1, 1)

        # Compute mean over channels to get a single heatmap
        heatmap = torch.mean(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)  # ReLU: only keep positive contributions

        # Normalize heatmap to [0, 1]
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()

def overlay_heatmap(img_tensor, heatmap, alpha=0.4):
    """
    Overlays the heatmap onto the original image.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape [1, 1, H, W] (grayscale).
        heatmap (np.ndarray): 2D heatmap with values in [0, 1].
        alpha (float): Transparency level of heatmap overlay.

    Returns:
        overlayed (np.ndarray): RGB image with heatmap overlay.
    """
    # Convert tensor to 2D numpy array (grayscale image)
    img = img_tensor.squeeze().cpu().numpy()

    # Normalize image to [0, 255] and convert to uint8
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(255 * img)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply color map (JET) to heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert BGR to RGB since OpenCV uses BGR by default
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Convert grayscale image to RGB for blending
    base_img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Overlay heatmap onto original image
    overlayed = cv2.addWeighted(heatmap_color, alpha, base_img_rgb, 1 - alpha, 0)
    return overlayed

def show_gradcam(model, image_tensor, class_idx=None):
    """
    Utility function to compute and display Grad-CAM heatmap.

    Args:
        model (torch.nn.Module): Trained CNN model.
        image_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].
        class_idx (int, optional): Target class index for Grad-CAM.
    """
    # Specify which layer to use for Grad-CAM. Adjust depending on model architecture.
    cam = GradCAM(model, target_layer=model.features[-3])
    heatmap = cam.generate(image_tensor, class_idx)
    overlay = overlay_heatmap(image_tensor, heatmap)

    # Display the result
    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()
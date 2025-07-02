import torch
import torch.nn as nn
import torch.nn.functional as F

class MedCNN(nn.Module):
    """
    A compact convolutional neural network for binary classification (e.g., pneumonia detection).
    Designed for grayscale chest X-ray images of size 224x224 with fewer parameters than MedCNN_Large.
    """

    def __init__(self, num_classes=2):
        """
        Initializes the MedCNN architecture.

        Args:
            num_classes (int): Number of output classes. Default is 2 (e.g., Normal and Pneumonia).
        """
        super(MedCNN, self).__init__()

        # === Convolutional feature extractor ===
        self.features = nn.Sequential(
            # Block 1: 1x224x224 → 32x112x112
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),                      # Downsample by 2
            nn.Dropout2d(p=0.1),                  # Regularization

            # Block 2: 32x112x112 → 64x56x56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),          # Downsample by 2
            nn.Dropout2d(p=0.1),

            # Block 3: 64x56x56 → 128x7x7
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),  # Force final spatial size to 7x7
            nn.Dropout2d(p=0.1)
        )

        # === Fully connected classifier ===
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # Flatten to 128*7*7
            nn.Linear(in_features=128 * 7 * 7, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),                       # Additional dropout for regularization
            nn.Linear(in_features=256, out_features=num_classes)  # Final logits
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 1, H, W]

        Returns:
            torch.Tensor: Logits of shape [B, num_classes]
        """
        x = self.features(x)        # Extract spatial features
        x = self.classifier(x)      # Classify
        return x
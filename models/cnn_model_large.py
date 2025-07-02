import torch
import torch.nn as nn
import torch.nn.functional as F

class MedCNN_Large(nn.Module):
    """
    A deep convolutional neural network for multi-label chest X-ray classification.
    This larger version uses more filters and layers compared to its small counterpart,
    making it more accurate but also computationally heavier.
    """

    def __init__(self, num_classes=14):
        """
        Initialize layers for MedCNN_Large model.

        Args:
            num_classes (int): Number of output classes. Default is 14 (NIH ChestX-ray14).
        """
        super(MedCNN_Large, self).__init__()

        # === Feature extractor ===
        self.features = nn.Sequential(
            # First conv block: input (1x224x224) → (64x224x224) → (64x224x224) → (64x112x112)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2

            # Second conv block: → (128x112x112) → (128x112x112) → (128x56x56)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third conv block: → (256x56x56) → (256x56x56) → (256x7x7)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Output spatial dims fixed to 7x7
        )

        # === Classifier (fully connected layers) ===
        self.classifier = nn.Sequential(
            nn.Flatten(),                             # Flatten 256x7x7 to 12544
            nn.Linear(256 * 7 * 7, 1024),             # Dense layer
            nn.ReLU(),
            nn.Dropout(0.5),                          # Regularization
            nn.Linear(1024, 512),                     # Hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),                          # Regularization
            nn.Linear(512, num_classes)              # Output layer (multi-label)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W]

        Returns:
            torch.Tensor: Output logits of shape [B, num_classes]
        """
        x = self.features(x)         # Extract hierarchical features
        return self.classifier(x)    # Classify using dense layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from models.cnn_model import MedCNN
from models.cnn_model_large import MedCNN_Large
from data.utils import get_dataloaders
from train.evaluate import evaluate_model
from utils.logger import Logger  # Custom logging utility

# Plot training and validation loss/accuracy curves
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path="training_plot.png"):
    """Generate and save training/validation loss and accuracy curves."""
    plt.figure(figsize=(10, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Core training loop for CNN with mixed precision and early stopping
def train(model, train_loader, val_loader, test_loader, device, logger, epochs=10, lr=1e-3, save_path='model.pt', patience=5):
    """
    Train a CNN model with early stopping, learning rate scheduling, and mixed-precision training.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler('cuda')  # For automatic mixed precision

    # Move model to device (CPU/GPU)
    model.to(device)

    # Track metrics across epochs
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate through training batches
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]')
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with AMP (mixed precision)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backpropagation and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            # Update progress bar
            loop.set_postfix(loss=running_loss / (loop.n + 1), acc=100. * correct / total)

        # Calculate training accuracy
        train_acc = 100. * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # Evaluate on validation set
        val_acc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                with autocast(device_type='cuda'):
                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Log results
        logger.info(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, "
                    f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        # Adjust learning rate if no improvement
        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved model checkpoint to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

    # Final evaluation on test set using best saved model
    model.load_state_dict(torch.load(save_path))
    test_acc = evaluate_model(model, test_loader, device)
    logger.info(f"✅ Final Test Accuracy (best model): {test_acc:.2f}%")
    print(f"✅ Final Test Accuracy (best model): {test_acc:.2f}%")

    # Save plots
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    logger.info("📈 Saved training/validation loss and accuracy plot to training_plot.png")

# Entry point for training script
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train the MedCNN model on chest X-ray dataset")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--model_size', type=str, required=True, help='Model size: [small | large]')
    args = parser.parse_args()

    # Validate model size argument
    if args.model_size not in ('small', 'large'):
        raise ValueError("Model size must be small or large")

    # Set computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize logger
    logger = Logger(log_file='train.log')
    logger.info("Initialized training script")

    # Load training, validation, and test data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, img_size=224)
    logger.info("Loaded data loaders")

    # Instantiate model
    if args.model_size == 'small':
        model = MedCNN(num_classes=2)
    else:
        model = MedCNN_Large(num_classes=2)
    logger.info("Initialized MedCNN model")

    # Start training
    train(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
          device=device, logger=logger, epochs=args.epochs, lr=1e-3,
          save_path='medcnn_chestxray.pt', patience=args.patience)
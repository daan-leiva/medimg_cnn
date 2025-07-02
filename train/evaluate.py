import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()
    
    all_preds = []   # Store all predicted class indices
    all_labels = []  # Store all true class labels

    # Disable gradient calculation for evaluation (saves memory and computation)
    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to target device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass through the model
            outputs = model(images)

            # Get predicted class (index of max logit)
            _, preds = torch.max(outputs, 1)

            # Move predictions and labels to CPU and store as NumPy arrays
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print confusion matrix and classification report
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))

    # Compute and return overall accuracy as a percentage
    acc = 100.0 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return acc

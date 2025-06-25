import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))

    acc = 100.0 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return acc
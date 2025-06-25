import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

def get_dataloaders(batch_size=32, img_size=224, data_dir="data/raw/chest_xray", val_split=0.1):
    # Augmentations only for training
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Standard transform for val/test
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    # Use full dataset first with train transform
    full_train_dataset = ImageFolder(root=train_path, transform=train_transform)

    # Split into train and val
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Apply test transform to val set
    val_dataset.dataset.transform = test_transform

    # Test set
    test_dataset = ImageFolder(root=test_path, transform=test_transform)

    # calculate the number of workers
    num_workers = min(4, os.cpu_count() // 2)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
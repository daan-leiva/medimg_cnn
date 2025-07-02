import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# Returns transformation pipeline for training images with augmentations
def get_train_transform(img_size=224):
    """
    Define training image preprocessing steps with data augmentation.
    Args:
        img_size (int): Target image size for resizing.
    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),              # Convert to grayscale (1 channel)
        transforms.Resize((img_size, img_size)),                  # Resize to uniform size
        transforms.RandomHorizontalFlip(),                        # Random horizontal flip
        transforms.RandomRotation(degrees=10),                    # Random rotation for robustness
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small random translations
        transforms.ToTensor(),                                    # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])               # Normalize pixel values
    ])
    return train_transform

# Returns transformation pipeline for validation and test images
def get_test_transform(img_size=224):
    """
    Define deterministic preprocessing steps for validation and test.
    Args:
        img_size (int): Target image size for resizing.
    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),              # Convert to grayscale
        transforms.Resize((img_size, img_size)),                  # Resize to uniform size
        transforms.ToTensor(),                                    # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])               # Normalize pixel values
    ])
    return test_transform

# Returns DataLoaders for training, validation, and testing
def get_dataloaders(batch_size=32, img_size=224, data_dir="data/raw/chest_xray", val_split=0.1):
    """
    Create PyTorch DataLoaders for training, validation, and test datasets.

    Args:
        batch_size (int): Number of images per batch.
        img_size (int): Size to which images will be resized.
        data_dir (str): Root directory of the chest X-ray dataset.
        val_split (float): Proportion of training data to use for validation.
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_train_transform(img_size)
    test_transform = get_test_transform(img_size)

    # Define paths
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    # Load full training set with augmentations
    full_train_dataset = ImageFolder(root=train_path, transform=train_transform)

    # Split into train and validation sets
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Override val dataset transform to use deterministic preprocessing
    val_dataset.dataset.transform = test_transform

    # Load test dataset
    test_dataset = ImageFolder(root=test_path, transform=test_transform)

    # Set number of worker threads for data loading
    num_workers = min(4, os.cpu_count() // 2)  # Conservative parallelism

    # Create data loaders with shuffling for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
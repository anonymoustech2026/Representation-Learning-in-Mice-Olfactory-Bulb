"""Dataset loading utilities."""

import os
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from torchvision import datasets
from .preprocessing import get_default_odor_transforms


def load_image_datasets(
    train_path: str,
    test_path: str,
    transform=None,
    val_path: Optional[str] = None
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, Optional[datasets.ImageFolder]]:
    """
    Load ImageFolder datasets.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        transform: Transform to apply to images
        val_path: Optional path to validation data
        
    Returns:
        Tuple of (train_dataset, test_dataset, val_dataset)
    """
    if transform is None:
        transform = get_default_odor_transforms()
    
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    val_dataset = None
    if val_path and os.path.exists(val_path):
        val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    
    return train_dataset, test_dataset, val_dataset


def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
    val_dataset: Optional[datasets.ImageFolder] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders from datasets.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset  
        val_dataset: Optional validation dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, test_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, test_loader, val_loader

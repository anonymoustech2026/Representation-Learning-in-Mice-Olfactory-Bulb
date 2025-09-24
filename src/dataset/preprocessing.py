"""Data preprocessing and transformation utilities."""

from typing import Dict, Any, Optional
from torchvision import transforms


def get_grayscale_transforms(
    mean: float = 0.5,
    std: float = 0.5,
    additional_transforms: Optional[list] = None
) -> transforms.Compose:
    """
    Get transforms for grayscale images.
    
    Args:
        mean: Normalization mean for grayscale
        std: Normalization std for grayscale  
        additional_transforms: Optional list of additional transforms
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ]
    
    # Add any additional transforms before normalization
    if additional_transforms:
        transform_list = transform_list[:-2] + additional_transforms + transform_list[-2:]
    
    return transforms.Compose(transform_list)


def get_default_odor_transforms() -> transforms.Compose:
    """Get default transforms for odor dataset (grayscale images)."""
    return get_grayscale_transforms(mean=0.5, std=0.5)

"""Visualization utilities for plotting training curves, confusion matrices, and data samples."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Dict, Tuple


def plot_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_path: str = 'results/confusion_matrix.png',
    figsize: Tuple[int, int] = (12, 10),
    threshold: float = 1.0,
    title: str = 'Confusion Matrix (Values in Percentage)'
) -> np.ndarray:
    """
    Generate and plot confusion matrix.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size
        threshold: Threshold for displaying values (values below this are hidden)
        title: Plot title
        
    Returns:
        Confusion matrix as numpy array
    """
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store true and predicted labels
    true_labels = []
    predictions = []
    
    # Get predictions
    with torch.no_grad():
        for data, target in dataloader:
            # Move tensors to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass and get predictions
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # Append true labels and predictions to lists
            true_labels.extend(target.view_as(predicted).cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
    
    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate the percentage of each cell relative to the row sum
    cm_percentage = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    cm_percentage *= 100  # Convert to percentage
    
    # Create annotation array: formatted percentages, but empty string for values below threshold
    annot_array = np.array([
        [f"{value:.2f}" if value > threshold else "" for value in row] 
        for row in cm_percentage
    ])
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Use class names if provided, otherwise use indices
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    # Create heatmap
    sns.heatmap(
        cm_percentage,
        annot=annot_array,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    save_path: str = 'results/training_curves.png',
    title: str = "Training Progress"
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        train_accuracies: Optional list of training accuracies
        val_accuracies: Optional list of validation accuracies
        save_path: Path to save the plot
        title: Plot title
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Determine subplot layout
    has_accuracy = train_accuracies is not None
    has_validation = val_losses is not None
    
    if has_accuracy:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if has_validation:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies if available
    if has_accuracy:
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        if val_accuracies is not None:
            ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Training curves saved to: {save_path}")
    plt.show()


def visualize_sample_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    num_samples: int = 16,
    cols: int = 4,
    save_path: str = 'results/sample_predictions.png',
    title: str = "Sample Predictions"
) -> None:
    """
    Visualize sample predictions with true and predicted labels.
    
    Args:
        model: Trained model
        dataloader: DataLoader for samples
        device: Device to run prediction on
        class_names: List of class names
        num_samples: Number of samples to display
        cols: Number of columns in the grid
        save_path: Path to save the plot
        title: Plot title
    """
    model.eval()
    
    # Get a batch of data
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu()
    
    # Calculate grid dimensions
    rows = (num_samples + cols - 1) // cols
    
    # Create the plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Handle grayscale images
        image = images[idx]
        if image.shape[0] == 1:  # Grayscale
            image = image.squeeze(0)
            ax.imshow(image, cmap='gray')
        else:  # RGB
            image = image.permute(1, 2, 0)
            ax.imshow(image)
        
        # Get labels
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        
        if class_names:
            true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
        else:
            true_name = f"Class {true_label}"
            pred_name = f"Class {pred_label}"
        
        # Color the title based on correctness
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Sample predictions saved to: {save_path}")
    plt.show()


def get_class_names_from_dataset(dataset) -> List[str]:
    """
    Extract class names from ImageFolder dataset.
    
    Args:
        dataset: ImageFolder dataset
        
    Returns:
        List of class names
    """
    if hasattr(dataset, 'classes'):
        return dataset.classes
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
        return dataset.dataset.classes
    else:
        return None

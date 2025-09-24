"""ROC curve analysis for multi-class classification."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Tuple


def compute_roc_curves(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Compute ROC curves for multi-class classification.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Tuple of (fpr_dict, tpr_dict, roc_auc_dict)
    """
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    # Get predictions and probabilities
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    true_labels = np.array(all_targets)
    predictions_probs = np.array(all_predictions)
    
    # Binarize the output labels for multi-class ROC
    true_labels_binarized = label_binarize(true_labels, classes=range(num_classes))
    
    # If binary classification, reshape
    if num_classes == 2:
        true_labels_binarized = np.column_stack([1 - true_labels_binarized, true_labels_binarized])
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], predictions_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        true_labels_binarized.ravel(), 
        predictions_probs.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc


def plot_multiclass_roc(
    fpr: Dict,
    tpr: Dict,
    roc_auc: Dict,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_path: str = 'results/roc_curves.png',
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Multi-Class ROC with Micro and Macro-Averaging',
    max_classes_display: int = 10
) -> None:
    """
    Plot multi-class ROC curves.
    
    Args:
        fpr: Dictionary of false positive rates
        tpr: Dictionary of true positive rates
        roc_auc: Dictionary of ROC AUC scores
        num_classes: Number of classes
        class_names: Optional list of class names
        save_path: Path to save the plot
        figsize: Figure size
        title: Plot title
        max_classes_display: Maximum number of individual class curves to display
    """
    plt.figure(figsize=figsize)
    
    # Define colors
    colors = cycle([
        'aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple',
        'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow',
        'black', 'lime', 'indigo', 'maroon', 'navy', 'silver', 'teal'
    ])
    
    # Plot individual class ROC curves (limit to max_classes_display)
    classes_to_plot = min(num_classes, max_classes_display)
    
    if classes_to_plot < num_classes:
        print(f"Displaying {classes_to_plot} out of {num_classes} class ROC curves for clarity.")
    
    for i, color in zip(range(classes_to_plot), colors):
        class_name = class_names[i] if class_names else f'Class {i}'
        plt.plot(
            fpr[i], tpr[i], 
            color=color, 
            lw=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
        color='deeppink', 
        linestyle=':', 
        linewidth=4
    )
    
    # Plot macro-average ROC curve
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
        color='navy', 
        linestyle=':', 
        linewidth=4
    )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"ROC curves saved to: {save_path}")
    plt.show()


def print_roc_summary(
    roc_auc: Dict,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Print ROC AUC summary statistics.
    
    Args:
        roc_auc: Dictionary of ROC AUC scores
        num_classes: Number of classes
        class_names: Optional list of class names
    """
    print("\n" + "="*50)
    print("ROC AUC SUMMARY")
    print("="*50)
    
    # Print per-class AUC scores
    print("\nPer-Class AUC Scores:")
    print("-" * 30)
    
    auc_scores = []
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        auc_score = roc_auc[i]
        auc_scores.append(auc_score)
        print(f"{class_name:>15}: {auc_score:.4f}")
    
    # Print summary statistics
    print("-" * 30)
    print(f"{'Micro-average':>15}: {roc_auc['micro']:.4f}")
    print(f"{'Macro-average':>15}: {roc_auc['macro']:.4f}")
    print(f"{'Mean AUC':>15}: {np.mean(auc_scores):.4f}")
    print(f"{'Std AUC':>15}: {np.std(auc_scores):.4f}")
    print(f"{'Min AUC':>15}: {np.min(auc_scores):.4f}")
    print(f"{'Max AUC':>15}: {np.max(auc_scores):.4f}")
    print("="*50)


def analyze_roc_performance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_path: str = 'results/roc_curves.png'
) -> Dict:
    """
    Complete ROC analysis pipeline.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional list of class names
        save_path: Path to save the ROC plot
        
    Returns:
        Dictionary containing ROC analysis results
    """
    print("Computing ROC curves...")
    fpr, tpr, roc_auc = compute_roc_curves(model, dataloader, device, num_classes, class_names)
    
    print("Plotting ROC curves...")
    plot_multiclass_roc(fpr, tpr, roc_auc, num_classes, class_names, save_path)
    
    print_roc_summary(roc_auc, num_classes, class_names)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'micro_auc': roc_auc['micro'],
        'macro_auc': roc_auc['macro']
    }

"""Detailed metrics calculation including precision, recall, F1-score for multi-class classification."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from typing import List, Optional, Dict, Tuple


def calculate_detailed_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate detailed metrics (precision, recall, F1, accuracy) for each class.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Dictionary containing detailed metrics
    """
    # Initialize dictionary to monitor metrics for each class
    class_metrics = {i: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for i in range(num_classes)}
    
    # Set model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            # Move tensors to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            
            # Store for sklearn metrics
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update confusion matrix elements for each sample
            for i in range(data.size(0)):
                true_label = target[i].item()
                pred_label = pred[i].item()
                
                # Update metrics for each class
                for class_idx in range(num_classes):
                    if class_idx == true_label:
                        if class_idx == pred_label:
                            class_metrics[class_idx]["TP"] += 1  # True positive
                        else:
                            class_metrics[class_idx]["FN"] += 1  # False negative
                    else:
                        if class_idx == pred_label:
                            class_metrics[class_idx]["FP"] += 1  # False positive
                        else:
                            class_metrics[class_idx]["TN"] += 1  # True negative
    
    # Calculate metrics for each class
    detailed_metrics = {}
    
    for class_idx, stats in class_metrics.items():
        # Calculate precision, recall, F1, and accuracy
        precision = stats["TP"] / (stats["TP"] + stats["FP"]) if stats["TP"] + stats["FP"] > 0 else 0
        recall = stats["TP"] / (stats["TP"] + stats["FN"]) if stats["TP"] + stats["FN"] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = (stats["TP"] + stats["TN"]) / (stats["TP"] + stats["TN"] + stats["FP"] + stats["FN"]) if (stats["TP"] + stats["TN"] + stats["FP"] + stats["FN"]) > 0 else 0
        
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        
        detailed_metrics[class_idx] = {
            'class_name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': stats["TP"] + stats["FN"],  # Total actual samples for this class
            'confusion_matrix': stats
        }
    
    # Calculate macro and micro averages using sklearn
    precision_scores, recall_scores, f1_scores, support_counts = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, zero_division=0
    )
    
    # Macro averages (unweighted mean)
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    macro_f1 = np.mean(f1_scores)
    
    # Micro averages (weighted by support)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='micro', zero_division=0
    )
    
    # Weighted averages (weighted by support)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    
    # Overall accuracy
    overall_accuracy = sum(np.array(all_targets) == np.array(all_predictions)) / len(all_targets)
    
    return {
        'per_class_metrics': detailed_metrics,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'overall_accuracy': overall_accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'support_counts': support_counts.tolist()
    }


def print_detailed_metrics(
    metrics: Dict,
    show_confusion_matrix: bool = False
) -> None:
    """
    Print detailed metrics in a formatted table.
    
    Args:
        metrics: Dictionary from calculate_detailed_metrics
        show_confusion_matrix: Whether to show confusion matrix elements
    """
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*80)
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Support':<10}")
    print("-" * 80)
    
    for class_idx, class_metrics in metrics['per_class_metrics'].items():
        print(f"{class_metrics['class_name']:<20} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<10.4f} "
              f"{class_metrics['f1']:<10.4f} "
              f"{class_metrics['accuracy']:<10.4f} "
              f"{class_metrics['support']:<10}")
        
        if show_confusion_matrix:
            cm = class_metrics['confusion_matrix']
            print(f"    TP: {cm['TP']}, FP: {cm['FP']}, FN: {cm['FN']}, TN: {cm['TN']}")
    
    # Print summary metrics
    print("-" * 80)
    print(f"{'Macro Avg':<20} "
          f"{metrics['macro_avg']['precision']:<12.4f} "
          f"{metrics['macro_avg']['recall']:<10.4f} "
          f"{metrics['macro_avg']['f1']:<10.4f} "
          f"{'N/A':<10} "
          f"{sum(metrics['support_counts']):<10}")
    
    print(f"{'Micro Avg':<20} "
          f"{metrics['micro_avg']['precision']:<12.4f} "
          f"{metrics['micro_avg']['recall']:<10.4f} "
          f"{metrics['micro_avg']['f1']:<10.4f} "
          f"{'N/A':<10} "
          f"{sum(metrics['support_counts']):<10}")
    
    print(f"{'Weighted Avg':<20} "
          f"{metrics['weighted_avg']['precision']:<12.4f} "
          f"{metrics['weighted_avg']['recall']:<10.4f} "
          f"{metrics['weighted_avg']['f1']:<10.4f} "
          f"{'N/A':<10} "
          f"{sum(metrics['support_counts']):<10}")
    
    print("-" * 80)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print("="*80)


def generate_classification_report(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate sklearn classification report.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: Optional list of class names
        save_path: Optional path to save the report
        
    Returns:
        Classification report as string
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Generate classification report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    print("\nSKLEARN CLASSIFICATION REPORT:")
    print("="*60)
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n")
            f.write(report)
        print(f"Classification report saved to: {save_path}")
    
    return report


def analyze_per_class_performance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_report_path: Optional[str] = None
) -> Dict:
    """
    Complete per-class performance analysis pipeline.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional list of class names
        save_report_path: Optional path to save classification report
        
    Returns:
        Dictionary containing detailed metrics
    """
    print("Calculating detailed per-class metrics...")
    
    # Calculate detailed metrics
    metrics = calculate_detailed_metrics(model, dataloader, device, num_classes, class_names)
    
    # Print results
    print_detailed_metrics(metrics)
    
    # Generate sklearn classification report
    generate_classification_report(model, dataloader, device, class_names, save_report_path)
    
    return metrics

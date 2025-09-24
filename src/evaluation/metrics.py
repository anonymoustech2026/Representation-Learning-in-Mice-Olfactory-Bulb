"""Evaluation metrics and testing utilities"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate model performance on given dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize dictionary to monitor test accuracy for each class
    class_accuracy = {i: {"correct": 0, "total": 0} for i in range(num_classes)}
    
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
            
            # Store predictions and targets for confusion matrix
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Compare predictions to true label
            correct = pred.eq(target.data.view_as(pred))
            
            # Update accuracy counts for each class
            for i in range(data.size(0)):
                label = target.data[i].item()
                class_accuracy[label]["total"] += 1
                if correct[i]:
                    class_accuracy[label]["correct"] += 1
    
    # Calculate per-class accuracies
    per_class_acc = {}
    for class_idx, stats in class_accuracy.items():
        if stats["total"] > 0:
            accuracy = 100 * stats["correct"] / stats["total"]
            per_class_acc[class_idx] = accuracy
        else:
            per_class_acc[class_idx] = 0.0
    
    # Calculate overall accuracy
    overall_correct = sum([stats["correct"] for stats in class_accuracy.values()])
    overall_total = sum([stats["total"] for stats in class_accuracy.values()])
    overall_accuracy = 100 * overall_correct / overall_total if overall_total > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_acc,
        'class_stats': class_accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'total_correct': overall_correct,
        'total_samples': overall_total
    }


def print_evaluation_results(
    results: Dict,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Print formatted evaluation results.
    
    Args:
        results: Results dictionary from evaluate_model
        class_names: Optional list of class names
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Print per-class accuracies
    print("\nPer-Class Accuracies:")
    print("-" * 40)
    
    for class_idx, accuracy in results['per_class_accuracy'].items():
        stats = results['class_stats'][class_idx]
        
        if class_names and len(class_names) > class_idx:
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"
        
        if stats["total"] > 0:
            print(f'{class_name:>15}: {accuracy:5.1f}% ({stats["correct"]:3d}/{stats["total"]:3d})')
        else:
            print(f'{class_name:>15}: N/A (no samples)')
    
    # Print overall accuracy
    print("-" * 40)
    print(f'{"Overall":>15}: {results["overall_accuracy"]:5.1f}% '
          f'({results["total_correct"]:3d}/{results["total_samples"]:3d})')
    print("="*60)


def get_model_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[List[int], List[int], List[torch.Tensor]]:
    """
    Get model predictions and confidence scores.
    
    Args:
        model: Trained model
        dataloader: DataLoader for prediction
        device: Device to run prediction on
        
    Returns:
        Tuple of (predictions, targets, confidence_scores)
    """
    model.eval()
    
    predictions = []
    targets = []
    confidence_scores = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions and confidence scores
            confidence, pred = torch.max(torch.softmax(output, dim=1), 1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            confidence_scores.extend(confidence.cpu())
    
    return predictions, targets, confidence_scores


def calculate_top_k_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        k: Top-k value
        
    Returns:
        Top-k accuracy as percentage
    """
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get top-k predictions
            _, pred = output.topk(k, 1, True, True)
            pred = pred.t()
            correct_pred = pred.eq(target.view(1, -1).expand_as(pred))
            
            # Count correct predictions
            correct += correct_pred[:k].sum().item()
            total += target.size(0)
    
    return 100 * correct / total if total > 0 else 0

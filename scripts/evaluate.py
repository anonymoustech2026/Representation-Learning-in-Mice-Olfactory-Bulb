"""Comprehensive evaluation script for trained odor classification model."""

import torch
import yaml
import argparse
import os

from src.models.cnn import OdorCNN
from src.dataset.loaders import load_image_datasets, create_dataloaders
from src.evaluation.metrics import evaluate_model, print_evaluation_results
from src.evaluation.detailed_metrics import analyze_per_class_performance  
from src.evaluation.roc_analysis import analyze_roc_performance
from src.utils.visualization import plot_confusion_matrix, get_class_names_from_dataset


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained odor classification model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, default='results/trained_model.pth',
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("ODOR CLASSIFICATION MODEL EVALUATION")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = OdorCNN(num_classes=config['data']['num_classes'])
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {args.model_path}")
    
    # Load test data
    print("\nLoading test dataset...")
    _, test_dataset, _ = load_image_datasets(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path']
    )
    
    _, test_loader, _ = create_dataloaders(
        None, test_dataset, None,
        batch_size=config.get('dataloader', {}).get('batch_size', 16)
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Get class names and number of classes
    class_names = get_class_names_from_dataset(test_dataset)
    num_classes = config['data']['num_classes']
    
    print(f"Number of classes: {num_classes}")
    if class_names:
        print(f"Classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
    
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # 1. Basic accuracy metrics
    print("\n1. BASIC ACCURACY METRICS")
    print("-" * 40)
    results = evaluate_model(model, test_loader, device, num_classes, class_names)
    print_evaluation_results(results, class_names)
    
    # 2. Confusion Matrix
    print("\n2. CONFUSION MATRIX")
    print("-" * 40)
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(model, test_loader, device, class_names, save_path=cm_path)
    
    # 3. Detailed Per-Class Metrics
    print("\n3. DETAILED PER-CLASS METRICS")
    print("-" * 40)
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    detailed_metrics = analyze_per_class_performance(
        model, test_loader, device, num_classes, class_names,
        save_report_path=report_path
    )
    
    # 4. ROC Analysis
    print("\n4. ROC CURVE ANALYSIS")
    print("-" * 40)
    roc_path = os.path.join(args.output_dir, 'roc_curves.png')
    roc_results = analyze_roc_performance(
        model, test_loader, device, num_classes, class_names,
        save_path=roc_path
    )
    
    # 5. Summary Report
    print("\n5. EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Macro F1-Score: {detailed_metrics['macro_avg']['f1']:.4f}")
    print(f"Micro F1-Score: {detailed_metrics['micro_avg']['f1']:.4f}")
    print(f"Weighted F1-Score: {detailed_metrics['weighted_avg']['f1']:.4f}")
    print(f"Macro AUC: {roc_results['macro_auc']:.4f}")
    print(f"Micro AUC: {roc_results['micro_auc']:.4f}")
    
    # Save summary to file
    summary_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ODOR CLASSIFICATION MODEL - EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Number of classes: {num_classes}\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Overall Accuracy: {results['overall_accuracy']:.2f}%\n")
        f.write(f"Macro F1-Score: {detailed_metrics['macro_avg']['f1']:.4f}\n")
        f.write(f"Micro F1-Score: {detailed_metrics['micro_avg']['f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {detailed_metrics['weighted_avg']['f1']:.4f}\n")
        f.write(f"Macro AUC: {roc_results['macro_auc']:.4f}\n")
        f.write(f"Micro AUC: {roc_results['micro_auc']:.4f}\n")
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")
    print(f"Summary report: {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()

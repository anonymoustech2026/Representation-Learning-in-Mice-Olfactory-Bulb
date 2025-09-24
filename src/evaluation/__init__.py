"""Evaluation module for model testing and metrics."""

from .metrics import (
    evaluate_model,
    print_evaluation_results,
    get_model_predictions,
    calculate_top_k_accuracy
)
from .roc_analysis import (
    compute_roc_curves,
    plot_multiclass_roc,
    print_roc_summary,
    analyze_roc_performance
)
from .detailed_metrics import (
    calculate_detailed_metrics,
    print_detailed_metrics,
    generate_classification_report,
    analyze_per_class_performance
)

__all__ = [
    'evaluate_model',
    'print_evaluation_results', 
    'get_model_predictions',
    'calculate_top_k_accuracy',
    'compute_roc_curves',
    'plot_multiclass_roc',
    'print_roc_summary',
    'analyze_roc_performance',
    'calculate_detailed_metrics',
    'print_detailed_metrics',
    'generate_classification_report',
    'analyze_per_class_performance'
]

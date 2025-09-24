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

__all__ = [
    'evaluate_model',
    'print_evaluation_results', 
    'get_model_predictions',
    'calculate_top_k_accuracy',
    'compute_roc_curves',
    'plot_multiclass_roc',
    'print_roc_summary',
    'analyze_roc_performance'
]

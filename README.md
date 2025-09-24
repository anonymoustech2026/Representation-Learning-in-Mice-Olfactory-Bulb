# Representation-Learning-in-Mice-Olfactory-Bulb
Anonymous Repository for ICLR 2026 

[ColorVSolfaction.pdf](https://github.com/user-attachments/files/22523426/ColorVSolfaction.pdf)



### Prerequisites
- Python 3.9 or higher



### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/odor-classification.git
cd odor-classification
```

2. **Set up Python environment**
```bash
# Using conda (recommended)
conda create -n odor-classification python=3.9
conda activate odor-classification

# Or using venv
python -m venv odor-env
source odor-env/bin/activate  # On Windows: odor-env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate iclr-paper-env
```

4. **Set up data directory**
```bash
mkdir -p data/{train,test,val}
mkdir -p results
```

## 📁 Repository Structure

```
odor-classification/
├── src/                          # Source code modules
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   └── cnn.py               # CNN implementations
│   ├── dataset/                  # Data loading utilities
│   │   ├── __init__.py
│   │   ├── loaders.py           # Dataset loading
│   │   └── preprocessing.py     # Data transforms
│   ├── evaluation/               # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py           # Basic metrics
│   │   ├── detailed_metrics.py  # Precision/Recall/F1
│   │   └── roc_analysis.py      # ROC curves
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── visualization.py     # Plotting functions
│       └── grad_cam.py          # Grad-CAM visualization
├── scripts/                      # Executable scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── grad_cam_analysis.py     # Grad-CAM analysis
├── configs/                      # Configuration files
│   └── base_config.yaml         # Base configuration
├── data/                         # Dataset directory
│   ├── train/                   # Training data
│   ├── test/                    # Test data
│   └── val/                     # Validation data
├── results/                      # Output directory
├── hpc/                         # HPC job scripts
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
└── README.md                    # This file
```

## Usage

### Training

**Basic training:**
```bash
python scripts/train.py
```

**With custom configuration:**
```bash
python scripts/train.py --config configs/experiment_config.yaml
```

### Evaluation

**Evaluate trained model:**
```bash
python scripts/evaluate.py
```

**With custom model and output directory:**
```bash
python scripts/evaluate.py \
  --model_path results/my_model.pth \
  --output_dir results/evaluation \
  --config configs/base_config.yaml
```

### Grad-CAM Visualization

**Analyze model interpretability:**
```bash
python scripts/grad_cam_analysis.py
```

##  Evaluation Metrics

The framework provides comprehensive evaluation including:

- **Basic Metrics**: Accuracy, Loss
- **Per-Class Metrics**: Precision, Recall, F1-Score
- **Confusion Matrix**: Visual and numerical analysis
- **ROC Curves**: Micro, Macro, and per-class ROC analysis
- **Grad-CAM**: Visual explanation of model decisions

##  Output Files

After training and evaluation, you'll find:

```
results/
├── trained_model.pth              # Saved model weights
├── confusion_matrix.png           # Confusion matrix plot
├── roc_curves.png                # ROC curve analysis
├── classification_report.txt      # Detailed metrics report
├── evaluation_summary.txt         # Summary of all metrics
├── training_curves.png           # Training/validation curves
└── grad_cam_analysis/             # Grad-CAM visualizations
    ├── grad_cam_conv3_*.png
    ├── grad_cam_conv4_*.png
    └── grad_cam_conv5_*.png
```

<img width="97" height="97" alt="n_methylpiperidine 53" src="https://github.com/user-attachments/assets/37515f44-e288-4f7c-9cf8-69600c72f1cc" />


<img width="305" height="305" alt="n_methylpiperidine gc" src="https://github.com/user-attachments/assets/4c3b7b2e-5607-4b33-b58c-fc74ccb9695b" />



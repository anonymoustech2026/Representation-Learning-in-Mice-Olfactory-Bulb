# Representation-Learning-in-Mice-Olfactory-Bulb
Anonymous Repository for ICLR 2026 

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







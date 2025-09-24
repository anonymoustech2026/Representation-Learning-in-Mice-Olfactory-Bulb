"""Script for Grad-CAM visualization analysis."""

import torch
import yaml
from src.models.cnn import OdorCNN
from src.utils.grad_cam import visualize_model_attention

# Load config and model
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = OdorCNN(num_classes=config['data']['num_classes'])
model.load_state_dict(torch.load('results/trained_model.pth'))

# Sample images for analysis
sample_images = [
    'path/to/sample1.jpg',
    'path/to/sample2.jpg',
    # Add your image paths
]

# Run comprehensive Grad-CAM analysis
visualize_model_attention(
    model=model,
    sample_images=sample_images,
    target_layers=['conv3', 'conv4', 'conv5'],
    save_dir='results/grad_cam_analysis'
)

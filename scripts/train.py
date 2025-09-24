""" training code"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.models.cnn import OdorCNN  # or create_model for factory
from src.dataset.loaders import load_image_datasets, create_dataloaders

# Load config
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup model, loss, optimizer, device
num_classes = config['data']['num_classes']
model = OdorCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")
print(f"Model: {model.__class__.__name__}")

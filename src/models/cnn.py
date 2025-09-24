"""CNN model for odor classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OdorCNN(nn.Module):
    """
    Convolutional Neural Network for odor classification.
    
    Architecture:
    - 5 convolutional layers with batch normalization
    - Max pooling after each conv layer
    - 3 fully connected layers with dropout
    """
    
    def __init__(
        self,
        num_classes: int = 35,
        input_channels: int = 1,
        dropout_rate: float = 0.3,
        input_size: int = 256  # Assuming 256x256 input images
    ):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale)
            dropout_rate: Dropout probability
            input_size: Input image size (assuming square images)
        """
        super(OdorCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # After 5 pooling operations: input_size // (2^5) = input_size // 32
        final_size = input_size // 32
        fc_input_size = 1024 * final_size * final_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional layers with batch norm, ReLU, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Extract feature maps from a specific layer.
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from
            
        Returns:
            Feature maps from specified layer
        """
        features = {}
        
        # Forward pass with feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        features['conv1'] = x
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        features['conv2'] = x
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        features['conv3'] = x
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        features['conv4'] = x
        
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        features['conv5'] = x
        
        return features.get(layer_name, x)


def create_odor_cnn(config: dict) -> OdorCNN:
    """
    Create CNN model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized CNN model
    """
    model_config = config.get('model', {})
    
    return OdorCNN(
        num_classes=config['data']['num_classes'],
        input_channels=model_config.get('input_channels', 1),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        input_size=model_config.get('input_size', 256)
    )

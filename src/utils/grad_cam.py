"""Grad-CAM visualization utilities for model interpretability."""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, List, Dict, Any
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN visualization.
    
    Reference: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str = 'conv4',
        device: Optional[torch.device] = None
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The CNN model
            target_layer: Name of the target layer for visualization
            device: Device to run computations on
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device).eval()
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Hook handles
        self.forward_hook = None
        self.backward_hook = None
        
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        target_module = self._get_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output
            self.activations.retain_grad()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.forward_hook = target_module.register_forward_hook(forward_hook)
        self.backward_hook = target_module.register_backward_hook(backward_hook)
    
    def _get_target_layer(self) -> nn.Module:
        """Get the target layer module from the model."""
        if hasattr(self.model, self.target_layer):
            return getattr(self.model, self.target_layer)
        else:
            # Try to find the layer by name in named_modules
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    return module
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
    
    def generate_cam(
        self,
        image_path: str,
        class_idx: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM for a given image.
        
        Args:
            image_path: Path to the input image
            class_idx: Target class index (if None, uses predicted class)
            transform: Image preprocessing transforms
            save_path: Optional path to save the result
            
        Returns:
            Tuple of (heatmap_overlay, predicted_class, confidence)
        """
        # Default transform if not provided
        if transform is None:
            transform = self._get_default_transform()
        
        # Load and preprocess image
        original_img = Image.open(image_path)
        tensor = transform(original_img).unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model(tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        # Use predicted class if class_idx not specified
        if class_idx is None:
            class_idx = predicted_class
        
        # Backward pass for the target class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Check if gradients are available
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check if target layer is correct.")
        
        # Generate Grad-CAM
        heatmap = self._generate_heatmap()
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
        
        # Create overlay
        heatmap_overlay = self._create_overlay(original_img, heatmap_resized)
        
        # Save if path provided
        if save_path:
            self._save_visualization(
                original_img, heatmap_resized, heatmap_overlay, 
                predicted_class, confidence_score, save_path
            )
        
        return heatmap_overlay, predicted_class, confidence_score
    
    def _generate_heatmap(self) -> np.ndarray:
        """Generate heatmap from activations and gradients."""
        # Get the gradients of the output with respect to the feature maps
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU on top of the heatmap
        heatmap = torch.clamp(heatmap, min=0)
        
        # Normalize the heatmap
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        return heatmap
    
    def _create_overlay(self, original_img: Image.Image, heatmap: np.ndarray) -> np.ndarray:
        """Create overlay of heatmap on original image."""
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert PIL image to numpy array
        original_array = np.array(original_img.convert('RGB'))
        
        # Create overlay
        overlay = heatmap_colored * 0.4 + original_array * 0.6
        
        return overlay.astype(np.uint8)
    
    def _save_visualization(
        self,
        original_img: Image.Image,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        predicted_class: int,
        confidence: float,
        save_path: str
    ) -> None:
        """Save visualization with original image, heatmap, and overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nPredicted: Class {predicted_class}\nConfidence: {confidence:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
        print(f"Grad-CAM visualization saved to: {save_path}")
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default transform for grayscale odor images."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def visualize_batch(
        self,
        image_paths: List[str],
        save_dir: str = 'results/grad_cam',
        class_indices: Optional[List[int]] = None
    ) -> List[Tuple[np.ndarray, int, float]]:
        """
        Generate Grad-CAM for a batch of images.
        
        Args:
            image_paths: List of image paths
            save_dir: Directory to save results
            class_indices: Optional list of target class indices
            
        Returns:
            List of (heatmap_overlay, predicted_class, confidence) tuples
        """
        os.makedirs(save_dir, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths):
            target_class = class_indices[i] if class_indices else None
            save_path = os.path.join(save_dir, f'grad_cam_{i}_{os.path.basename(image_path)}')
            
            try:
                result = self.generate_cam(image_path, target_class, save_path=save_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((None, -1, 0.0))
        
        return results
    
    def __del__(self):
        """Clean up hooks when object is destroyed."""
        if self.forward_hook:
            self.forward_hook.remove()
        if self.backward_hook:
            self.backward_hook.remove()


def grad_cam_analysis(
    model: nn.Module,
    image_path: str,
    target_layer: str = 'conv4',
    class_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Convenience function for single image Grad-CAM analysis.
    
    Args:
        model: Trained CNN model
        image_path: Path to input image
        target_layer: Name of target layer
        class_idx: Target class index (if None, uses predicted)
        save_path: Optional save path
        device: Device to use
        
    Returns:
        Tuple of (heatmap_overlay, predicted_class, confidence)
    """
    grad_cam = GradCAM(model, target_layer, device)
    
    try:
        result = grad_cam.generate_cam(image_path, class_idx, save_path=save_path)
        return result
    finally:
        # Ensure cleanup
        del grad_cam


def visualize_model_attention(
    model: nn.Module,
    sample_images: List[str],
    target_layers: List[str] = ['conv3', 'conv4', 'conv5'],
    save_dir: str = 'results/grad_cam_analysis',
    class_names: Optional[List[str]] = None
) -> None:
    """
    Comprehensive visualization of model attention across multiple layers.
    
    Args:
        model: Trained model
        sample_images: List of sample image paths
        target_layers: List of layers to visualize
        save_dir: Directory to save results
        class_names: Optional class names
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for img_idx, image_path in enumerate(sample_images):
        print(f"Processing image {img_idx + 1}/{len(sample_images)}: {os.path.basename(image_path)}")
        
        for layer_name in target_layers:
            try:
                save_path = os.path.join(
                    save_dir, 
                    f'grad_cam_{layer_name}_img{img_idx}_{os.path.basename(image_path)}'
                )
                
                overlay, pred_class, confidence = grad_cam_analysis(
                    model, image_path, layer_name, save_path=save_path
                )
                
                class_name = class_names[pred_class] if class_names else f"Class {pred_class}"
                print(f"  {layer_name}: Predicted {class_name} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"  Error with {layer_name}: {e}")
        
        print("-" * 50)

"""Main training script for odor classification"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.models.cnn import OdorCNN
from src.dataset.loaders import load_image_datasets, create_dataloaders
from src.evaluation.metrics import evaluate_model, print_evaluation_results
from src.utils.visualization import plot_confusion_matrix, get_class_names_from_dataset
from src.evaluation.roc_analysis import analyze_roc_performance
from src.evaluation.detailed_metrics import analyze_per_class_performance

def main():
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
    
    # Create dataloaders
    train_dataset, test_dataset, val_dataset = load_image_datasets(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path']
    )
    train_loader, test_loader, val_loader = create_dataloaders(
        train_dataset, test_dataset, val_dataset,
        batch_size=config.get('dataloader', {}).get('batch_size', 16)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Training loop
    n_epochs = config.get('training', {}).get('epochs', 40)
    
    print(f"Starting training for {n_epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(1, n_epochs + 1):
        # Set model to training mode
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move tensors to GPU if available
            data, target = data.to(device), target.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * data.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Calculate average loss and accuracy
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total
        
        # Print training statistics
        print(f'Epoch: {epoch:2d}/{n_epochs} | '
              f'Training Loss: {avg_train_loss:.6f} | '
              f'Training Accuracy: {train_accuracy:.2f}%')
    
    print("-" * 50)
    print("Training completed!")
    
    # Save the trained model
    torch.save(model.state_dict(), 'results/trained_model.pth')
    print("Model saved to results/trained_model.pth")

    print("Evaluating model on test set...")
    test_results = evaluate_model(model, test_loader, device, num_classes)
    print_evaluation_results(test_results)

    print("Creating confusion matrix...")
    class_names = get_class_names_from_dataset(test_dataset)
    cm = plot_confusion_matrix(model, test_loader, device, class_names)

    print("Performing ROC analysis...")
    roc_results = analyze_roc_performance(model, test_loader, device, num_classes, class_names)

    print("Analyzing detailed per-class performance...")
    detailed_metrics = analyze_per_class_performance(
        model, test_loader, device, num_classes, class_names,
        save_report_path='results/classification_report.txt'
    )

if __name__ == "__main__":
    main()

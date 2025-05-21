#!/usr/bin/env python3
# bold_evaluate_model.py - Script to evaluate the trained BOLD transformer model

import numpy as np
import torch
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the BOLD transformer model module for accessing classes
from bold_transformer_model import BOLDTransformerModel, BOLDDataset, Config, plot_confusion_matrix, load_joint_data, parse_bold_annotations, prepare_dataset

def load_model(model_path):
    """
    Load the trained model from disk
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model: Loaded model
        config: Model configuration
        label_encoder_classes: Class names for prediction
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Add safe globals for numpy arrays
    import torch.serialization
    torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
    
    # Load the model with weights_only=False
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Get configuration
    config = checkpoint['config']
    
    # Create model with same parameters
    model = BOLDTransformerModel(
        num_classes=config['num_classes'],
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create label encoder with the same classes
    label_encoder_classes = checkpoint['label_encoder_classes']
    
    return model, config, label_encoder_classes

def evaluate_on_dataset(model, dataset_path, annotation_file, label_encoder_classes, max_samples=None):
    """
    Evaluate the model on a new dataset
    
    Args:
        model: The loaded model
        dataset_path: Path to the dataset (joint data)
        annotation_file: Path to the annotation file
        label_encoder_classes: The class labels from the trained model
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        accuracy: Evaluation accuracy
        predictions: Model predictions
        targets: Ground truth labels
    """
    # Parse annotations
    samples = parse_bold_annotations(annotation_file, max_samples)
    
    # Prepare dataset
    X, y = prepare_dataset(samples)
    
    # Create a label encoder with the same classes as training
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_classes)
    
    # Create dataset
    test_dataset = BOLDDataset(X, y, label_encoder)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE
    )
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Evaluation
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() * 100
    
    return accuracy, all_preds, all_targets

def main():
    # Load model
    model_path = 'SUMMER/bold_transformer_model.pt'
    model, config, label_encoder_classes = load_model(model_path)
    
    # Dataset paths
    test_annotations = os.path.join(Config.BOLD_ANNOTATIONS_DIR, "test.csv")
    
    # Evaluate on test set if it exists
    if os.path.exists(test_annotations):
        print(f"Evaluating model on test dataset: {test_annotations}")
        accuracy, preds, targets = evaluate_on_dataset(model, Config.BOLD_JOINTS_DIR, test_annotations, label_encoder_classes)
        
        # Convert numeric labels back to class names
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(label_encoder_classes)
        class_names = label_encoder.classes_
        
        # Print results
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        print("Confusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(targets, preds, target_names=class_names))
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, title='Test Confusion Matrix')
        plt.savefig('SUMMER/bold_test_confusion_matrix.png')
        print("Confusion matrix saved to SUMMER/bold_test_confusion_matrix.png")
    else:
        print(f"Test annotation file not found: {test_annotations}")
        print("Using validation set from the original training data instead")
        # In this case, we'd use the validation results from the training

if __name__ == "__main__":
    main() 
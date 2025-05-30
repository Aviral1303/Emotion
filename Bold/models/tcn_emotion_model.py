#!/usr/bin/env python3
# tcn_emotion_model.py - Emotion classification from motion capture data using TCN with PyTorch

# --- 1. IMPORT LIBRARIES ---
import numpy as np
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter
import time

# Import BVH library
try:
    import bvh
except ImportError:
    print("Installing bvh library...")
    import subprocess
    subprocess.check_call(["pip", "install", "bvh"])
    import bvh

# --- 2. DATA LOADING ---
def parse_filename(filename):
    """
    Extract metadata from BVH filenames.
    
    Args:
        filename: Name of the BVH file (e.g., 'F01A0V1.bvh')
        
    Returns:
        Dictionary containing actor_id, emotion
    """
    # Extract just the filename without path or extension
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Map emotion codes to emotion names
    emotion_code_map = {
        'A': 'Angry',
        'D': 'Disgust',
        'F': 'Fearful',
        'H': 'Happy',
        'N': 'Neutral',
        'SA': 'Sad',
        'SU': 'Surprise'
    }
    
    # Handle different possible file naming patterns
    if len(base_filename) >= 6:
        # For format F01A0V1 or M01A0V1
        actor_id = base_filename[:3]  # F01 or M01
        
        # Extract emotion code (could be 1 or 2 characters)
        if base_filename[3:5] in ['SA', 'SU']:
            emotion_code = base_filename[3:5]
        else:
            emotion_code = base_filename[3]
        
        emotion = emotion_code_map.get(emotion_code, 'Unknown')
        
        return {
            "actor_id": actor_id,
            "emotion": emotion,
            "filename": os.path.basename(filename)
        }
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")

def load_bvh_files(data_dir, limit=150):
    """Load BVH files from the dataset directory, limited to specified number"""
    # Find all BVH files in the dataset directory and subdirectories
    all_bvh_files = []
    actor_dirs = glob.glob(os.path.join(data_dir, "BVH", "*"))
    
    for actor_dir in actor_dirs:
        bvh_files = glob.glob(os.path.join(actor_dir, "*.bvh"))
        all_bvh_files.extend(bvh_files)
    
    # Limit to the specified number of files
    all_bvh_files = all_bvh_files[:limit]
    
    # Create tuples of (file_path, emotion)
    bvh_files_with_emotions = []
    for file_path in all_bvh_files:
        try:
            metadata = parse_filename(file_path)
            emotion = metadata["emotion"]
            bvh_files_with_emotions.append((file_path, emotion))
        except Exception as e:
            print(f"Error processing filename {file_path}: {e}")
    
    print(f"Loaded {len(bvh_files_with_emotions)} BVH files")
    return bvh_files_with_emotions

# --- 3. DATA PREPROCESSING ---
def extract_joint_data(bvh_file, seq_len=150, num_joints=20):
    """Extract joint positions from BVH file"""
    try:
        # Parse BVH file using the bvh library
        with open(bvh_file, 'r') as f:
            mocap = bvh.Bvh(f.read())
        
        # Extract joint positions
        frames = []
        for frame_idx in range(min(seq_len, mocap.nframes)):
            joint_positions = []
            # Use available joints (up to num_joints)
            joint_names = mocap.get_joints_names()[:num_joints]
            
            for joint_name in joint_names:
                try:
                    # Try to get position channels
                    pos = mocap.frame_joint_channels(frame_idx, joint_name, ['Xposition', 'Yposition', 'Zposition'])
                except:
                    # If position channels don't exist, use zeros as placeholder
                    pos = [0, 0, 0]  # In a real implementation, calculate from rotations
                joint_positions.extend(pos)
            
            # Ensure we have exactly 60 features (20 joints × 3 coords)
            if len(joint_positions) < 60:
                joint_positions.extend([0] * (60 - len(joint_positions)))
            elif len(joint_positions) > 60:
                joint_positions = joint_positions[:60]
                
            frames.append(joint_positions)
        
        # Pad sequences if needed to reach seq_len
        if len(frames) < seq_len:
            last_frame = frames[-1] if frames else [0] * 60
            frames.extend([last_frame] * (seq_len - len(frames)))
                
    except Exception as e:
        print(f"Error extracting joint data from {bvh_file}: {e}")
        # Return empty array of correct shape
        return np.zeros((seq_len, 60))
    
    return np.array(frames)

def normalize_positions(joint_data):
    """
    Normalize joint positions by subtracting mean and dividing by standard deviation.
    This helps the model converge faster and improves generalization.
    
    Args:
        joint_data: Motion data of shape [seq_len, features]
        
    Returns:
        Normalized joint data
    """
    # Calculate mean and std across all joints and timeframes for this sample
    mean = np.mean(joint_data)
    std = np.std(joint_data)
    
    # Normalize (add small epsilon to avoid division by zero)
    normalized_data = (joint_data - mean) / (std + 1e-6)
    
    return normalized_data

def preprocess_data(bvh_files_with_emotions):
    """Process all BVH files and extract features and labels"""
    X = []  # Features (joint positions)
    y = []  # Labels (emotions)
    
    for file_path, emotion in bvh_files_with_emotions:
        # Extract and normalize joint data
        try:
            joint_data = extract_joint_data(file_path)
            joint_data = normalize_positions(joint_data)
            
            X.append(joint_data)
            y.append(emotion)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(X), np.array(y)

# --- 4. TCN MODEL DEFINITION ---
class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer, equivalent to TF/Keras' Conv1D with padding='causal'
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        # Input shape: [batch, channels, seq_len]
        x = self.conv(x)
        # Remove future timesteps (right padding)
        return x[:, :, :-self.padding] if self.padding else x

class TCNBlock(nn.Module):
    """TCN block with dilated causal convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(TCNBlock, self).__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class TCNEmotionClassifier(nn.Module):
    """
    TCN model for emotion classification using PyTorch.
    """
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(TCNEmotionClassifier, self).__init__()

        # Model parameters
        filters = 64
        kernel_size = 5
        
        # TCN layers (need to transpose input from [batch, seq_len, features] to [batch, features, seq_len])
        self.tcn_block1 = TCNBlock(input_dim, filters, kernel_size, dilation=1, dropout=dropout)
        self.tcn_block2 = TCNBlock(filters, filters, kernel_size, dilation=2, dropout=dropout)
        
        # Global average pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers
        self.dense1 = nn.Linear(filters, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Transpose from [batch, seq_len, features] to [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Apply TCN blocks
        x = self.tcn_block1(x)
        x = self.tcn_block2(x)
        
        # Global average pooling
        x = self.pool(x).squeeze(-1)
        
        # Dense layers
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        
        return logits

# --- EARLY STOPPING ---
class EarlyStopping:
    """Early stopping handler to monitor validation loss"""
    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = -val_loss  # Higher score is better (negative loss)
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        # Save model state
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Save as a file too
        torch.save(model.state_dict(), 'best_tcn_model.pt')
        self.val_loss_min = val_loss
        
    def restore_best_weights(self, model):
        """Restore the best model weights"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print('Restored best model weights')

# --- 5. TRAINING AND EVALUATION FUNCTIONS ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        _, targets_cls = torch.max(targets, 1)
        total += targets.size(0)
        correct += (predicted == targets_cls).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            _, targets_cls = torch.max(targets, 1)
            total += targets.size(0)
            correct += (predicted == targets_cls).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets_cls.cpu().numpy())
    
    val_loss = running_loss / total
    val_acc = 100 * correct / total
    return val_loss, val_acc, np.array(all_preds), np.array(all_targets)

# Function to plot training history
def plot_training_history(history, save_path='tcn_training_plot.png'):
    """
    Plot and save training/validation accuracy and loss curves.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")
    plt.close()

# --- 6. MAIN FUNCTION ---
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_dir = "kinematic-dataset-of-actors-expressing-emotions-2.1.0"  # Path to the dataset folder
    
    try:
        # Load BVH files with emotion labels
        bvh_files = load_bvh_files(data_dir, limit=150)  # Load 150 files
        
        if not bvh_files:
            print("No BVH files found. Please check the dataset path.")
            return
        
        # Preprocess data
        X, y = preprocess_data(bvh_files)
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
        
        # Print class distribution
        class_counts = Counter(y)
        print("\nClass distribution:")
        for emotion, count in class_counts.items():
            print(f"  {emotion}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        print(f"\nEmotions found: {label_encoder.classes_} ({num_classes} classes)")
        
        # One-hot encode
        y_onehot = np.zeros((y_encoded.size, num_classes))
        y_onehot[np.arange(y_encoded.size), y_encoded] = 1
        
        # Train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Print split sizes
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Define model parameters
        input_dim = X.shape[2]  # 60 features
        
        # Create model
        model = TCNEmotionClassifier(input_dim=input_dim, num_classes=num_classes)
        model.to(device)
        
        # Model summary
        print("\nModel Architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        
        # Define loss function and optimizer with L2 weight decay
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=5, verbose=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Train model
        print("\nTraining model...")
        epochs = 50
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation phase
            val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Check early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Restore best model weights
        early_stopping.restore_best_weights(model)
        
        # Training time
        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f} seconds")
        
        # Plot training history
        plot_training_history(history)
        
        # Final evaluation
        test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Confusion matrix calculation (counts)
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(y_true)):
            confusion[y_true[i]][y_pred[i]] += 1
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("Rows: True labels, Columns: Predicted labels")
        print("Labels:", label_encoder.classes_)
        print(confusion)
        
        # Show sample predictions
        print("\nSample predictions:")
        for i in range(min(5, len(y_true))):
            true_emotion = label_encoder.classes_[y_true[i]]
            pred_emotion = label_encoder.classes_[y_pred[i]]
            print(f"Sample {i}: True = {true_emotion}, Predicted = {pred_emotion}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
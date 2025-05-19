#!/usr/bin/env python3
# transformer_emotion_model.py - Emotion classification from motion capture data using Transformers

# --- 1. IMPORT LIBRARIES ---
import numpy as np
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from sklearn.metrics import classification_report, precision_recall_fscore_support
import copy  # Added for deepcopy

# Import BVH library the same way the SUMMER code does
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

def load_bvh_files(data_dir, limit=100):
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

def augment_underrepresented_classes(X, y, class_names, target_count=None):
    """
    Augment underrepresented classes using simple time-based transformations.
    
    Args:
        X: Motion data of shape [n_samples, seq_len, features]
        y: Labels of shape [n_samples]
        class_names: List of class names
        target_count: Target count for each class (if None, use majority class count)
        
    Returns:
        Augmented data (X_aug, y_aug)
    """
    # Count samples in each class
    class_counts = {}
    for class_name in class_names:
        class_counts[class_name] = np.sum(y == class_name)
    
    # If no target_count provided, use the majority class count
    if target_count is None:
        target_count = max(class_counts.values())
    
    print(f"Augmenting data to {target_count} samples per class")
    
    # Initialize augmented data lists
    X_aug = []
    y_aug = []
    
    # Process each class
    for class_name in class_names:
        # Get indices for this class
        indices = np.array([i for i, label in enumerate(y) if label == class_name])
        
        # No augmentation needed if already at or above target count
        if len(indices) >= target_count:
            # Take exactly target_count samples
            selected_indices = indices[:target_count]
            X_class = X[selected_indices]
            y_class = np.array([class_name] * len(selected_indices))
            X_aug.append(X_class)
            y_aug.append(y_class)
            continue
        
        # Calculate how many augmented samples needed
        n_augment = target_count - len(indices)
        print(f"  Adding {n_augment} augmented samples for class '{class_name}'")
        
        # Original samples
        X_class = list(X[indices])
        y_class = [class_name] * len(indices)
        
        # Create augmented samples with more aggressive augmentation
        for j in range(n_augment):
            # Randomly select a sample to augment
            idx = np.random.choice(indices)
            sample = X[idx].copy()
            
            # Apply multiple augmentation techniques for more diversity
            # Randomly choose 2-3 augmentations to apply
            num_augmentations = np.random.randint(2, 4)
            augmentation_types = np.random.choice([
                'time_shift', 'time_scale', 'noise', 'reverse_segments', 'interpolate', 'mix_up'
            ], num_augmentations, replace=False)
            
            for augmentation_type in augmentation_types:
                if augmentation_type == 'time_shift':
                    # Shift in time (roll)
                    shift = np.random.randint(1, 30)  # Increased shift range
                    sample = np.roll(sample, shift, axis=0)
                    
                elif augmentation_type == 'time_scale':
                    # Time scaling (stretch or compress)
                    scale_factor = np.random.uniform(0.8, 1.2)  # Wider scale range
                    seq_len, n_features = sample.shape
                    
                    # Interpolate to new length
                    new_len = int(seq_len * scale_factor)
                    if new_len < seq_len:
                        # Stretch
                        indices_interp = np.linspace(0, new_len-1, seq_len).astype(int)
                        stretched = np.zeros((new_len, n_features))
                        for k in range(new_len):
                            stretched[k] = sample[min(k, seq_len-1)]
                        sample = stretched[indices_interp]
                    else:
                        # Compress
                        indices_interp = np.linspace(0, seq_len-1, new_len).astype(int)
                        compressed = sample[indices_interp]
                        # Get a slice of compressed of size seq_len
                        start = np.random.randint(0, max(1, new_len - seq_len))
                        sample = compressed[start:start+seq_len]
                        # Pad if needed
                        if sample.shape[0] < seq_len:
                            pad_size = seq_len - sample.shape[0]
                            sample = np.pad(sample, ((0, pad_size), (0, 0)), 'edge')
                    
                elif augmentation_type == 'noise':
                    # Add small noise
                    noise_level = np.random.uniform(0.02, 0.08)  # Increased noise range
                    noise = np.random.normal(0, noise_level, sample.shape)
                    sample = sample + noise
                    
                elif augmentation_type == 'reverse_segments':
                    # Reverse short segments
                    n_segments = np.random.randint(3, 8)  # More segments
                    seq_len = sample.shape[0]
                    segment_size = seq_len // n_segments
                    
                    for k in range(n_segments):
                        if np.random.random() < 0.4:  # Increased chance to reverse
                            start = k * segment_size
                            end = min((k + 1) * segment_size, seq_len)
                            sample[start:end] = sample[start:end][::-1]
                            
                elif augmentation_type == 'interpolate':
                    # Interpolate between two samples of the same class
                    if len(indices) > 1:
                        idx2 = np.random.choice([i for i in indices if i != idx])
                        sample2 = X[idx2]
                        alpha = np.random.uniform(0.3, 0.7)  # Wider interpolation range
                        sample = alpha * sample + (1 - alpha) * sample2
                
                elif augmentation_type == 'mix_up':
                    # Mix with a sample from another class but with lower weight
                    other_indices = np.array([i for i, label in enumerate(y) if label != class_name])
                    if len(other_indices) > 0:
                        other_idx = np.random.choice(other_indices)
                        other_sample = X[other_idx].copy()
                        # Use high weight for the original class (0.7-0.9)
                        mix_weight = np.random.uniform(0.7, 0.9)
                        sample = mix_weight * sample + (1 - mix_weight) * other_sample
            
            # Normalize augmented sample
            sample = normalize_positions(sample)
            
            # Add augmented sample
            X_class.append(sample)
            y_class.append(class_name)
        
        # Convert back to arrays and append
        X_aug.append(np.array(X_class))
        y_aug.append(np.array(y_class))
    
    # Concatenate all classes
    X_aug = np.concatenate(X_aug, axis=0)
    y_aug = np.concatenate(y_aug, axis=0)
    
    return X_aug, y_aug

# --- 4. POSITIONAL ENCODING ---
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""
    def __init__(self, d_model, max_seq_len=150):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]

# --- 5. MODEL DEFINITION ---
class TransformerEmotionClassifier(nn.Module):
    """Enhanced Transformer encoder model for emotion classification with better handling of rare classes"""
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, dim_feedforward=512, dropout=0.3, num_layers=3):
        super().__init__()
        
        # Feature projection
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)  # Add dropout after projection layer
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layer with higher dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,  # Increased from the default
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Class-specific features with separate attention heads for each class
        self.class_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
            for _ in range(num_classes)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Final classification with class-specific pathways
        self.class_pathways = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(num_classes)
        ])
        
        # Additional specialized pathways for the positive class (index 2)
        self.positive_pathway = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Neutral-specific pathway with extra capacity
        self.neutral_pathway = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Class query embeddings for attention (learnable parameters)
        self.class_queries = nn.Parameter(torch.randn(1, num_classes, d_model))
        
    def forward(self, x):
        # Project features
        x = self.feature_projection(x)
        x = self.activation(x)
        x = self.dropout1(x)  # Apply dropout
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Initialize output tensor
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, 3).to(x.device)
        
        # Class-specific attention and pathways
        for i in range(len(self.class_attention)):
            # Get class-specific query
            query = self.class_queries[:, i:i+1, :].repeat(batch_size, 1, 1)
            
            # Apply class-specific attention
            attended_output, _ = self.class_attention[i](
                query,  # queries
                x,      # keys
                x       # values
            )
            
            # Mean pooling of attended output
            pooled = torch.mean(attended_output, dim=1)  # [B, d_model]
            
            # Apply class-specific pathway
            if i == 1:  # For Neutral class (index 1)
                outputs[:, i] = self.neutral_pathway(pooled).squeeze(-1)
            elif i == 2:  # For Positive class (index 2)
                outputs[:, i] = self.positive_pathway(pooled).squeeze(-1)
            else:
                outputs[:, i] = self.class_pathways[i](pooled).squeeze(-1)
        
        return outputs

# --- 6. TRAINING FUNCTIONS ---
def train_model(model, dataloader, criterion, optimizer, device):
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
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, targets_cls = torch.max(targets.data, 1)
        total += targets.size(0)
        correct += (predicted == targets_cls).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

def evaluate_model(model, dataloader, criterion, device, threshold=None):
    """
    Evaluate model performance with optional soft decision threshold
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the data
        criterion: Loss function
        device: Device to run evaluation on
        threshold: Optional dictionary of class-specific thresholds
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Apply softmax to get probabilities
            probs = nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Apply normal max prediction if no threshold is provided
            if threshold is None:
                _, predicted = torch.max(outputs.data, 1)
            else:
                # Apply soft decision threshold
                predicted = apply_soft_threshold(probs, threshold)
                
            _, targets_cls = torch.max(targets.data, 1)
            total += targets.size(0)
            correct += (predicted == targets_cls).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets_cls.cpu().numpy())
    
    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy, all_preds, all_targets, all_probs

def apply_soft_threshold(probs, threshold):
    """
    Apply soft decision thresholding to probabilities
    
    Args:
        probs: Probabilities from model output (after softmax) [batch_size, num_classes]
        threshold: Dictionary of class-specific thresholds, e.g., {1: 0.4} for class 1
        
    Returns:
        Tensor of predicted classes
    """
    # Start with standard predictions (argmax)
    _, predicted = torch.max(probs, 1)
    
    # Apply thresholds for specific classes
    for class_idx, thresh in threshold.items():
        # If probability exceeds threshold, assign to this class
        mask = probs[:, class_idx] >= thresh
        predicted[mask] = class_idx
    
    # Special handling for the positive class (class 2) - if it's close enough to the max
    # and no other threshold triggered, consider it positive
    if 2 not in threshold:  # Only if we don't already have a threshold for positive
        positive_boost_threshold = 0.85  # If positive class prob is 85% of max, boost it
        max_probs, _ = torch.max(probs, dim=1)
        positive_close_mask = (probs[:, 2] >= positive_boost_threshold * max_probs) & (predicted != 1)
        predicted[positive_close_mask] = 2
        
    return predicted

def print_classification_metrics(y_true, y_pred, target_names):
    """Print detailed classification metrics"""
    # Get precision, recall, f1-score
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print("\nDetailed Classification Metrics:")
    print("Class             Precision    Recall       F1-Score    Support")
    print("----------------------------------------------------------------")
    
    for i, class_name in enumerate(target_names):
        print(f"{class_name:<15} {precision[i]:.4f}       {recall[i]:.4f}       {f1[i]:.4f}       {support[i]}")
    
    # Also get macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print("----------------------------------------------------------------")
    print(f"Macro Average    {precision_macro:.4f}       {recall_macro:.4f}       {f1_macro:.4f}       {len(y_true)}")
    
    # Print full classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

# --- EARLY STOPPING ---
class EarlyStopping:
    """
    Early stopping handler to monitor training progress and stop when no improvement is seen.
    Saves the best model weights based on validation loss.
    """
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
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        # Save model state
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.val_loss_min = val_loss
        
    def restore_best_weights(self, model):
        """Restore the best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print('Restored best model weights')

# --- EMOTION GROUPING ---
def group_emotions(emotions, group_method='detailed'):
    """
    Group emotions into broader categories for better classification performance.
    
    Args:
        emotions: List of emotion labels
        group_method: Grouping method ('detailed'=7 classes, 'broad'=3 classes)
        
    Returns:
        List of grouped emotion labels
    """
    if group_method == 'detailed':
        # Keep original 7 classes
        return emotions
    elif group_method == 'broad':
        # Group into 3 broader categories
        emotion_groups = {
            'Positive': ['Happy', 'Surprise'],
            'Negative': ['Angry', 'Fearful', 'Sad', 'Disgust'],
            'Neutral': ['Neutral']
        }
        
        grouped_emotions = []
        for emotion in emotions:
            for group, members in emotion_groups.items():
                if emotion in members:
                    grouped_emotions.append(group)
                    break
        
        return grouped_emotions
    else:
        raise ValueError(f"Unknown grouping method: {group_method}")

# Function to plot training history
def plot_training_history(history, save_path='training_plot.png'):
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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create a more visually appealing colormap
    cmap = plt.cm.Blues
    
    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, pad=20)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Add labels and ticks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.tight_layout()
    
    # Add a grid to make it easier to read
    plt.grid(False)
    
    # Add class distribution information
    class_totals = cm.sum(axis=1)
    for i, total in enumerate(class_totals):
        plt.text(len(classes), i, f'Total: {int(total)}',
                ha='left', va='center', fontsize=10)
    
    return plt

def train_cv(X, y_encoded, y_onehot, num_classes, device, k_folds=5):
    """
    Train the model with k-fold cross-validation
    
    Args:
        X: Input features
        y_encoded: Encoded labels (integers)
        y_onehot: One-hot encoded labels
        num_classes: Number of classes
        device: Device to run training on
        k_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation results
    """
    # Define the K-fold cross-validator
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Lists to store fold results
    fold_train_accs = []
    fold_val_accs = []
    fold_models = []
    
    print(f"\nPerforming {k_folds}-fold cross-validation...")
    
    # K-fold Cross Validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y_encoded)):
        print(f"\nFold {fold+1}/{k_folds}")
        
        # Sample data for this fold
        X_train_fold = X[train_ids]
        y_train_fold = y_onehot[train_ids]
        X_val_fold = X[val_ids]
        y_val_fold = y_onehot[val_ids]
        
        # Convert to tensors
        X_train_fold = torch.FloatTensor(X_train_fold)
        y_train_fold = torch.FloatTensor(y_train_fold)
        X_val_fold = torch.FloatTensor(X_val_fold)
        y_val_fold = torch.FloatTensor(y_val_fold)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Define model
        seq_len = X.shape[1]
        input_dim = X.shape[2]
        model = TransformerEmotionClassifier(input_dim, num_classes)
        model.to(device)
        
        # Use custom weights
        custom_weights = torch.FloatTensor([0.2, 0.7, 0.3])  # [Negative, Neutral, Positive]
        class_weights = custom_weights.to(device)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower learning rate
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=5, verbose=True)
        
        # Soft decision thresholds to improve balance between classes
        # Lower threshold for Neutral class to increase likelihood of prediction
        soft_threshold = {1: 0.15}  # Lower threshold for Neutral class (index 1)
        print(f"Using soft threshold for Neutral class: {soft_threshold[1]}")
        
        # Train the model
        epochs = 30
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            
            # Validation phase
            val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device, threshold=soft_threshold)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Monitor class-specific performance every 5 epochs
            if (epoch + 1) % 5 == 0:
                _, _, train_preds, train_true, _ = evaluate_model(model, train_loader, criterion, device, threshold=soft_threshold)
                print(f"\nFold {fold+1}, Epoch {epoch+1} - Class-specific performance (training set):")
                unique_classes = np.unique(train_true)
                for i in unique_classes:
                    true_pos = sum((np.array(train_true) == i) & (np.array(train_preds) == i))
                    all_true = sum(np.array(train_true) == i)
                    all_pred = sum(np.array(train_preds) == i)
                    
                    recall = true_pos / max(1, all_true) * 100
                    precision = true_pos / max(1, all_pred) * 100
                    
                    print(f"  Class {i}: Recall={recall:.1f}%, Precision={precision:.1f}%, "
                          f"Samples={all_true}, Predictions={all_pred}")
            
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
        
        # Evaluate final performance
        _, train_acc, _, _, _ = evaluate_model(model, train_loader, criterion, device, threshold=soft_threshold)
        _, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device, threshold=soft_threshold)
        
        # Store results
        fold_train_accs.append(train_acc)
        fold_val_accs.append(val_acc)
        fold_models.append(copy.deepcopy(model.state_dict()))
        
        print(f"Fold {fold+1} - Final Train Acc: {train_acc:.2f}%, Final Val Acc: {val_acc:.2f}%")
    
    # Calculate cross-validation metrics
    cv_results = {
        'train_acc_mean': np.mean(fold_train_accs),
        'train_acc_std': np.std(fold_train_accs),
        'val_acc_mean': np.mean(fold_val_accs),
        'val_acc_std': np.std(fold_val_accs),
        'fold_models': fold_models,
        'fold_train_accs': fold_train_accs,
        'fold_val_accs': fold_val_accs
    }
    
    print("\nCross-Validation Results:")
    print(f"Training Accuracy: {cv_results['train_acc_mean']:.2f}% ± {cv_results['train_acc_std']:.2f}%")
    print(f"Validation Accuracy: {cv_results['val_acc_mean']:.2f}% ± {cv_results['val_acc_std']:.2f}%")
    
    return cv_results

# --- 7. MAIN FUNCTION ---
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_dir = "kinematic-dataset-of-actors-expressing-emotions-2.1.0"  # Updated path to the dataset folder
    
    # Whether to use broader emotion categories (reduces to 3 classes)
    use_broad_emotions = True  # Set to False for original 7 emotion classes
    
    try:
        # Load BVH files with emotion labels
        bvh_files = load_bvh_files(data_dir, limit=50)
        
        if not bvh_files:
            print("No BVH files found. Please check the dataset path.")
            return
        
        # Preprocess data
        X, y = preprocess_data(bvh_files)
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
        
        # Group emotions if needed
        if use_broad_emotions:
            y = group_emotions(y, 'broad')
            print("Using broad emotion categories (3 classes)")
        
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
        
        # Resample the data to make it more balanced using augmentation
        # Target a larger count for better representation
        negative_count = class_counts['Negative']
        target_count = negative_count  # Use the majority class count
        X_augmented, y_augmented = augment_underrepresented_classes(X, y, label_encoder.classes_, target_count)
        
        # Print augmented class distribution
        augmented_class_counts = Counter(y_augmented)
        print("\nAugmented class distribution:")
        for emotion, count in augmented_class_counts.items():
            print(f"  {emotion}: {count} samples ({count/len(y_augmented)*100:.1f}%)")
        
        # Use the augmented data from now on
        X = X_augmented
        y = y_augmented
        
        # Encode labels for augmented data
        y_encoded = label_encoder.transform(y)
        
        # One-hot encode
        y_onehot = np.eye(num_classes)[y_encoded]
        
        # Stratified train-test split to maintain class distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y_encoded))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]
        y_encoded_train, y_encoded_test = y_encoded[train_idx], y_encoded[test_idx]
        
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
        
        # Use smaller batch size for better stochasticity
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Define model parameters
        seq_len = X.shape[1]  # 150
        input_dim = X.shape[2]  # 60
            
        # Create model
        model = TransformerEmotionClassifier(input_dim, num_classes)
        model.to(device)
        
        # Calculate class weights to address class imbalance
        # Higher weights for minority classes
        custom_weights = torch.FloatTensor([0.2, 0.4, 0.6])  # [Negative, Neutral, Positive]
        class_weights = custom_weights.to(device)
        print(f"Using custom class weights: {class_weights}")
        
        # Define loss function with class weights and optimizer with L2 weight decay
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)  # Lower learning rate
        
        # Use scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Soft decision thresholds to improve balance between classes
        # Adjust thresholds for all classes
        soft_threshold = {
            1: 0.20,  # Lower threshold for Neutral class
            2: 0.15   # Lower threshold for Positive class to increase predictions
        }
        print(f"Using soft threshold for Neutral class: {soft_threshold[1]}")
        print(f"Using soft threshold for Positive class: {soft_threshold[2]}")
        
        # Model summary
        print(f"\nModel Architecture:")
        print(model)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Early stopping
        early_stopping = EarlyStopping(patience=15, verbose=True)  # Increased patience
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Train model with increased epochs
        print("\nTraining model...")
        epochs = 150  # Increased for better convergence
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            
            # Validation phase
            val_loss, val_acc, _, _, _ = evaluate_model(model, test_loader, criterion, device, threshold=soft_threshold)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
                
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
            # Monitor class-specific performance every 5 epochs
            if (epoch + 1) % 5 == 0:
                _, _, train_preds, train_true, _ = evaluate_model(model, train_loader, criterion, device, threshold=soft_threshold)
                print("\nClass-specific performance (training set):")
                unique_classes = np.unique(train_true)
                for i in unique_classes:
                    true_pos = sum((np.array(train_true) == i) & (np.array(train_preds) == i))
                    all_true = sum(np.array(train_true) == i)
                    all_pred = sum(np.array(train_preds) == i)
                    
                    recall = true_pos / max(1, all_true) * 100
                    precision = true_pos / max(1, all_pred) * 100
                    
                    class_name = label_encoder.classes_[i] if i < len(label_encoder.classes_) else f"Class {i}"
                    print(f"  {class_name}: Recall={recall:.1f}%, Precision={precision:.1f}%, "
                          f"Samples={all_true}, Predictions={all_pred}")
                
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
        # Get predictions for training set with soft threshold
        train_loss, train_acc, train_pred, train_true, train_probs = evaluate_model(
            model, train_loader, criterion, device, threshold=soft_threshold
        )
        print(f"\nFinal Training Accuracy: {train_acc:.2f}%")
        
        # Get predictions for test set with soft threshold
        test_loss, test_acc, test_pred, test_true, test_probs = evaluate_model(
            model, test_loader, criterion, device, threshold=soft_threshold
        )
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
        
        # Confusion matrix for training set
        train_confusion = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(train_true)):
            train_confusion[train_true[i]][train_pred[i]] += 1
        
        # Confusion matrix for test set
        test_confusion = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(test_true)):
            test_confusion[test_true[i]][test_pred[i]] += 1
        
        # Print confusion matrices
        print("\nTraining Confusion Matrix:")
        print("Rows: True labels, Columns: Predicted labels")
        print("Labels:", label_encoder.classes_)
        print(train_confusion)
        
        print("\nTest Confusion Matrix:")
        print("Rows: True labels, Columns: Predicted labels")
        print("Labels:", label_encoder.classes_)
        print(test_confusion)
        
        # Print detailed classification metrics
        print("\nTraining Set Metrics:")
        print_classification_metrics(train_true, train_pred, label_encoder.classes_)
        
        print("\nTest Set Metrics:")
        print_classification_metrics(test_true, test_pred, label_encoder.classes_)
        
        # Plot confusion matrices
        plt.figure(figsize=(15, 6))

        # Training confusion matrix
        plt.subplot(1, 2, 1)
        plot_confusion_matrix(train_confusion, label_encoder.classes_, 
                             title='Training Confusion Matrix\n(Accuracy: {:.2f}%)'.format(train_acc * 100))

        # Test confusion matrix
        plt.subplot(1, 2, 2)
        plot_confusion_matrix(test_confusion, label_encoder.classes_,
                             title='Test Confusion Matrix\n(Accuracy: {:.2f}%)'.format(test_acc * 100))

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Show sample predictions
        print("\nSample predictions:")
        for i in range(min(5, len(test_true))):
            true_emotion = label_encoder.classes_[test_true[i]]
            pred_emotion = label_encoder.classes_[test_pred[i]]
            print(f"Sample {i}: True = {true_emotion}, Predicted = {pred_emotion}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
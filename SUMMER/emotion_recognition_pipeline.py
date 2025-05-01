#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion Recognition from Motion Data Pipeline

This script implements a research pipeline for emotion recognition using motion capture data.
The pipeline includes data loading, preprocessing, tokenization, model training, and evaluation.

Dataset: Kinematic Dataset of Actors Expressing Emotions
         https://physionet.org/content/kinematic-actors-emotions/1.0.0/
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse

try:
    import bvh
except ImportError:
    print("Installing bvh library...")
    import subprocess
    subprocess.check_call(["pip", "install", "bvh"])
    import bvh

# Global configurations
EMOTION_MAP = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Fearful": 2,  # Map to the same ID as "Fear"
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6
}

# Reverse mapping for readability
EMOTION_LABELS = {v: k for k, v in EMOTION_MAP.items()}

# Default configuration
CONFIG = {
    "data_dir": "./data",
    "output_dir": "./output",
    "sequence_length": 100,  # Target length after tokenization
    "embedding_dim": 64,     # Dimension for embeddings
    "batch_size": 32,
    "epochs": 20,            # Increased from 5 for better convergence
    "learning_rate": 0.001,
    "test_size": 0.2,
    "validation_size": 0.1,
    "use_class_weights": True,  # Use class weights for imbalanced data
    "feature_engineering": True, # Add velocity features
    "seed": 42
}


def setup_directories(config: Dict) -> None:
    """
    Create necessary directories for data storage and outputs.
    
    Args:
        config: Configuration dictionary containing directory paths
    """
    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    

def parse_filename(filename: str) -> Dict:
    """
    Extract metadata from BVH filenames.
    
    Args:
        filename: Name of the BVH file (e.g., 'F01A0V1.bvh')
        
    Returns:
        Dictionary containing actor_id, emotion, and take_id
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
            scenario_version = base_filename[5:]
        else:
            emotion_code = base_filename[3]
            scenario_version = base_filename[4:]
        
        emotion = emotion_code_map.get(emotion_code, 'Unknown')
        
        # Extract the version at the end (take_id)
        take_id = int(scenario_version[-1]) if scenario_version[-1].isdigit() else 1
        
        return {
            "actor_id": actor_id,
            "emotion": emotion,
            "emotion_id": EMOTION_MAP.get(emotion, -1),  # Map to numerical ID or -1 if not found
            "take_id": take_id,
            "filename": os.path.basename(filename)
        }
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")


def parse_bvh_file(filepath: str, selected_joints: Optional[List[str]] = None) -> np.ndarray:
    """
    Parse a BVH file and extract joint positions over time.
    
    Args:
        filepath: Path to the BVH file
        selected_joints: Optional list of joint names to include (if None, use all)
        
    Returns:
        Array of shape [timesteps, num_joints*3] containing joint positions
    """
    with open(filepath) as f:
        mocap = bvh.Bvh(f.read())
    
    # Get number of frames and joints
    num_frames = mocap.nframes
    joints = list(mocap.get_joints())
    num_joints = len(joints)
    
    # Initialize the array for positions
    positions = np.zeros((num_frames, num_joints, 3))
    
    # Extract joint positions for each frame
    for frame_idx in range(num_frames):
        for joint_idx, joint in enumerate(joints):
            # Get the joint position for this frame
            try:
                x, y, z = mocap.frame_joint_position(frame_idx, joint)
                positions[frame_idx, joint_idx] = [x, y, z]
            except:
                # Some joints might not have positions
                positions[frame_idx, joint_idx] = [0, 0, 0]
    
    # If selected joints are specified, filter the data
    if selected_joints:
        joint_indices = [joints.index(joint) for joint in selected_joints 
                         if joint in joints]
        positions = positions[:, joint_indices, :]
    
    # Reshape to [timesteps, num_joints*3]
    num_frames, num_joints, dims = positions.shape
    positions_flat = positions.reshape(num_frames, num_joints * dims)
    
    return positions_flat


def normalize_motion_data(motion_data: np.ndarray) -> np.ndarray:
    """
    Normalize motion data by centering and scaling.
    
    Args:
        motion_data: Array of shape [timesteps, features]
        
    Returns:
        Normalized motion data
    """
    # Center the root joint position at each timestep
    # Assuming first 3 values are root joint x,y,z
    root_positions = motion_data[:, :3]
    centered_data = motion_data.copy()
    
    # For each timestep, subtract root position from all joint positions
    for i in range(0, motion_data.shape[1], 3):
        centered_data[:, i:i+3] -= root_positions
    
    # Global scaling to normalize the overall magnitude
    max_abs_val = np.max(np.abs(centered_data))
    if max_abs_val > 0:
        centered_data /= max_abs_val
    
    return centered_data


def tokenize_sequence_pooling(motion_data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Tokenize a motion sequence using average pooling to achieve fixed length.
    
    Args:
        motion_data: Array of shape [original_timesteps, features]
        target_length: Desired sequence length after tokenization
        
    Returns:
        Tokenized sequence of shape [target_length, features]
    """
    original_length = motion_data.shape[0]
    features = motion_data.shape[1]
    
    if original_length == target_length:
        return motion_data
    
    # Initialize output array
    tokenized = np.zeros((target_length, features))
    
    # For each target frame, compute the average of corresponding original frames
    for i in range(target_length):
        start_idx = int(i * original_length / target_length)
        end_idx = int((i + 1) * original_length / target_length)
        
        # Handle edge case for the last segment
        if end_idx > original_length:
            end_idx = original_length
            
        if start_idx == end_idx:
            # If only one frame maps to this segment, use it directly
            tokenized[i] = motion_data[start_idx]
        else:
            # Otherwise, take the average of all frames in the segment
            tokenized[i] = np.mean(motion_data[start_idx:end_idx], axis=0)
    
    return tokenized


def tokenize_sequence_pca(motion_data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Tokenize a motion sequence using PCA for dimensionality reduction.
    
    Args:
        motion_data: Array of shape [original_timesteps, features]
        target_length: Desired sequence length after tokenization
        
    Returns:
        Tokenized sequence of shape [target_length, features]
    """
    # First downsample to target length using pooling
    downsampled = tokenize_sequence_pooling(motion_data, target_length)
    
    # Apply PCA if the feature dimension is very high (optional further reduction)
    if downsampled.shape[1] > 100:  # If feature dimension is too high
        pca = PCA(n_components=min(downsampled.shape[1], 100))
        return pca.fit_transform(downsampled)
    
    return downsampled


class EmotionMotionDataset(Dataset):
    """
    PyTorch Dataset for motion-based emotion recognition.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            X: Motion data of shape [num_samples, sequence_length, features]
            y: Emotion labels of shape [num_samples]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """
    Simple MLP classifier for emotion recognition.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        """
        Initialize the MLP classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            num_classes: Number of emotion classes
        """
        super(MLPClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for emotion recognition.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        """
        Initialize the LSTM classifier.
        
        Args:
            input_dim: Dimension of input features per timestep
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of emotion classes
        """
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: [batch_size, sequence_length, input_dim]
        output, _ = self.lstm(x)
        # Use only the last time step output
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for emotion recognition.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, num_classes: int):
        """
        Initialize the Transformer classifier.
        
        Args:
            input_dim: Dimension of input features per timestep
            d_model: Dimension of transformer model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer encoder layers
            num_classes: Number of emotion classes
        """
        super(TransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: [batch_size, sequence_length, input_dim]
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        # Global average pooling over sequence length
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def calculate_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Calculate weights inversely proportional to class frequencies.
    
    Args:
        y: Array of class labels
        
    Returns:
        Tensor of class weights for balanced training
    """
    num_classes = max(len(EMOTION_LABELS), 7)  # Ensure at least 7 classes
    
    # Initialize weights with ones
    weights = np.ones(num_classes)
    
    # Count class frequencies
    class_counts = np.bincount(y, minlength=num_classes)
    n_samples = len(y)
    
    # Calculate weights for classes that have samples
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = n_samples / (num_classes * class_counts[i])
        else:
            weights[i] = 1.0  # Default weight for classes with no samples
    
    # Normalize weights to avoid extremely large values
    if np.max(weights) > 10:
        weights = weights / np.max(weights) * 10
    
    return torch.FloatTensor(weights)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                config: Dict) -> Tuple[nn.Module, Dict]:
    """
    Train an emotion recognition model.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        
    Returns:
        Trained model and training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get labels from train_loader for class weights calculation
    train_labels = train_loader.dataset.y.numpy()
    
    # Use class weights if enabled
    if config.get("use_class_weights", False):
        class_weights = calculate_class_weights(train_labels).to(device)
        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optim = optimizer.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', factor=0.5, patience=3
    )
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optim.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optim.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["output_dir"], "best_model.pt"))
            print(f"New best validation accuracy: {val_acc:.4f} - Model saved")
    
    return model, history


def plot_training_history(history: Dict, config: Dict) -> None:
    """
    Plot training and validation loss/accuracy.
    
    Args:
        history: Dictionary containing training history
        config: Configuration dictionary
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "training_history.png"))
    plt.close()


def process_dataset(data_dir: str, config: Dict) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Process the entire dataset.
    
    Args:
        data_dir: Directory containing BVH files
        config: Configuration dictionary
        
    Returns:
        Tokenized motion data, emotion labels, and metadata
    """
    # Find all BVH files
    bvh_files = glob.glob(os.path.join(data_dir, "*.bvh"))
    
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {data_dir}")
    
    print(f"Found {len(bvh_files)} BVH files")
    
    # Process each file
    all_motion_data = []
    all_labels = []
    metadata_list = []
    
    for bvh_file in tqdm(bvh_files, desc="Processing BVH files"):
        try:
            # Extract metadata from filename
            metadata = parse_filename(bvh_file)
            
            # Skip files with invalid emotion IDs
            if metadata["emotion_id"] == -1:
                print(f"Skipping {bvh_file} - Unknown emotion: {metadata['emotion']}")
                continue
                
            # Parse BVH file
            motion_data = parse_bvh_file(bvh_file)
            
            # Normalize motion data
            motion_data = normalize_motion_data(motion_data)
            
            # Apply feature engineering if enabled
            if config.get("feature_engineering", False):
                motion_data = enhance_motion_features(motion_data)
            
            # Tokenize sequence
            tokenized_data = tokenize_sequence_pooling(motion_data, config["sequence_length"])
            
            # Append to lists
            all_motion_data.append(tokenized_data)
            all_labels.append(metadata["emotion_id"])
            metadata_list.append(metadata)
            
        except Exception as e:
            print(f"Error processing {bvh_file}: {e}")
    
    # Convert lists to arrays
    X = np.array(all_motion_data)
    y = np.array(all_labels)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    
    # Print emotion distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Emotion distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {EMOTION_LABELS[label]}: {count} samples")
    
    return X, y, metadata_df


def extract_velocity_features(positions: np.ndarray) -> np.ndarray:
    """
    Calculate velocity features from position data.
    
    Args:
        positions: Array of shape [timesteps, num_joints*3]
        
    Returns:
        Array of velocity features with same shape
    """
    # First-order difference approximates velocity
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]
    return velocity


def extract_acceleration_features(positions: np.ndarray) -> np.ndarray:
    """
    Calculate acceleration features from position data.
    
    Args:
        positions: Array of shape [timesteps, num_joints*3]
        
    Returns:
        Array of acceleration features with same shape
    """
    # Second-order difference approximates acceleration
    velocity = extract_velocity_features(positions)
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def enhance_motion_features(motion_data: np.ndarray) -> np.ndarray:
    """
    Enhance motion data with derived features.
    
    Args:
        motion_data: Array of shape [timesteps, num_joints*3]
        
    Returns:
        Enhanced features with additional velocity and acceleration
    """
    velocity = extract_velocity_features(motion_data)
    acceleration = extract_acceleration_features(motion_data)
    
    # Scale to balance magnitude differences
    v_scale = 0.5
    a_scale = 0.25
    
    # Concatenate features along feature dimension
    enhanced = np.concatenate([
        motion_data, 
        velocity * v_scale, 
        acceleration * a_scale
    ], axis=1)
    
    return enhanced


def main():
    """Main function to run the entire pipeline."""
    # Setup directories
    setup_directories(CONFIG)
    
    # Check if data dir exists
    if not os.path.exists(CONFIG["data_dir"]):
        print(f"Data directory {CONFIG['data_dir']} not found. Please download the dataset and extract it to this directory.")
        print("Dataset URL: https://physionet.org/content/kinematic-actors-emotions/1.0.0/")
        return
    
    # Process dataset
    print("Processing dataset...")
    X, y, metadata_df = process_dataset(CONFIG["data_dir"], CONFIG)
    
    # Determine the number of samples
    n_samples = len(y)
    
    # Disable class weights for very small datasets (less than 50 samples)
    if n_samples < 50 and CONFIG.get("use_class_weights", False):
        print("Warning: Dataset too small (<50 samples). Disabling class weights.")
        CONFIG["use_class_weights"] = False
    
    # Save processed data
    print("Saving processed data...")
    np.save(os.path.join(CONFIG["output_dir"], "X.npy"), X)
    np.save(os.path.join(CONFIG["output_dir"], "y.npy"), y)
    metadata_df.to_csv(os.path.join(CONFIG["output_dir"], "metadata.csv"), index=False)
    
    # Split data based on dataset size and class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    
    # Check if dataset is too small for stratification
    min_test_size = n_classes  # Need at least one sample per class
    actual_test_size = int(n_samples * CONFIG["test_size"])
    
    if min_test_size > actual_test_size or np.min(counts) < 2:
        print("Warning: Dataset too small for stratified sampling. Using random split.")
        # For very small datasets, use simple random split
        test_size = max(0.2, min_test_size / n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=CONFIG["seed"]
        )
        
        # If there are enough samples, create validation set
        if len(X_train) > 2 * n_classes:
            val_size = CONFIG["validation_size"] / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=CONFIG["seed"]
            )
        else:
            # For extremely small datasets, use test set as validation
            print("Warning: Training set too small for separate validation. Using test set for validation.")
            X_val, y_val = X_test, y_test
    else:
        # Normal case: enough data for stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["seed"], stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=CONFIG["validation_size"]/(1-CONFIG["test_size"]),
            random_state=CONFIG["seed"], stratify=y_train
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create datasets and dataloaders
    train_dataset = EmotionMotionDataset(X_train, y_train)
    val_dataset = EmotionMotionDataset(X_val, y_val)
    test_dataset = EmotionMotionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
    
    # Create and train models
    print("Training MLP model...")
    input_dim = X_train.shape[1] * X_train.shape[2]  # sequence_length * features
    num_classes = len(EMOTION_MAP)  # Get actual number of classes from EMOTION_MAP
    
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    
    mlp_model = MLPClassifier(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
    mlp_model, mlp_history = train_model(mlp_model, train_loader, val_loader, CONFIG)
    
    # Plot training history
    plot_training_history(mlp_history, CONFIG)
    
    print("Training LSTM model...")
    lstm_model = LSTMClassifier(
        input_dim=X_train.shape[2],  # features
        hidden_dim=128,
        num_layers=2,
        num_classes=len(EMOTION_MAP)
    )
    lstm_model, lstm_history = train_model(lstm_model, train_loader, val_loader, CONFIG)
    
    print("Done!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Emotion Recognition from Motion Data Pipeline')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing BVH files')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    CONFIG["data_dir"] = args.data_dir
    CONFIG["output_dir"] = args.output_dir
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    
    main() 
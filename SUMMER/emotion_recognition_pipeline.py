#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion Recognition from Motion Data Pipeline using MotionBERT

This script implements a research pipeline for emotion recognition using motion capture data
with a MotionBERT-based model.

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
import torch.nn.functional as F
import json

try:
    import bvh
except ImportError:
    print("Installing bvh library...")
    import subprocess
    subprocess.check_call(["pip", "install", "bvh"])
    import bvh

from motion_bert_model import MotionBERTClassifier, create_motion_bert_config, train_motion_bert

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

# Important joints for emotion recognition
IMPORTANT_JOINTS = [
    "Hips",           # Root
    "Spine",         # Core
    "Spine1",
    "Spine2",
    "Neck",          # Upper body
    "Head",
    "LeftArm",       # Arms
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",     # Legs
    "LeftLeg",
    "LeftFoot",
    "RightUpLeg",
    "RightLeg",
    "RightFoot"
]

# Reverse mapping for readability
EMOTION_LABELS = {v: k for k, v in EMOTION_MAP.items()}

# Default configuration
CONFIG = {
    "data_dir": "./test_data",
    "output_dir": "./output",
    "sequence_length": 300,
    "batch_size": 16,
    "epochs": 150,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "test_size": 0.2,
    "validation_size": 0.2,
    "use_class_weights": True,
    "feature_engineering": True,
    "seed": 42,
    "dropout_rate": 0.2,
    "early_stopping_patience": 20,
    "use_data_augmentation": True,
    "warmup_steps": 2000,
    "max_steps": 20000,
    "gradient_clip_val": 1.0,
    
    # Feature extraction
    "use_velocity": True,
    "use_acceleration": True,
    "use_joint_angles": True,
    "use_selected_joints": True,
    "selected_joints": IMPORTANT_JOINTS,
    
    # Model architecture
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
    "hidden_dim": 256,
    "num_classes": 7,
    
    # Training enhancements
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    "augment_prob": 0.5,
    "use_focal_loss": True,
    "focal_gamma": 2.0
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
    Parse a BVH file and extract joint positions and rotations over time.
    
    Args:
        filepath: Path to the BVH file
        selected_joints: Optional list of joint names to include (if None, use all)
        
    Returns:
        Array of shape [timesteps, num_joints*6] containing joint positions and rotations
    """
    with open(filepath) as f:
        mocap = bvh.Bvh(f.read())
    
    # Get number of frames and joints
    num_frames = mocap.nframes
    joints = list(mocap.get_joints())
    
    # Filter joints if specified
    if selected_joints:
        joints = [j for j in joints if j in selected_joints]
    
    num_joints = len(joints)
    
    # Initialize the array for positions and rotations
    features = np.zeros((num_frames, num_joints, 6))  # 3 for position, 3 for rotation
    
    # Extract joint positions and rotations for each frame
    for frame_idx in range(num_frames):
        for joint_idx, joint in enumerate(joints):
            try:
                # Get both position and rotation channels
                x, y, z = mocap.frame_joint_channels(frame_idx, joint, ['Xposition', 'Yposition', 'Zposition'])
                rx, ry, rz = mocap.frame_joint_channels(frame_idx, joint, ['Xrotation', 'Yrotation', 'Zrotation'])
                features[frame_idx, joint_idx] = [x, y, z, rx, ry, rz]
            except Exception as e:
                # Use previous frame's values if available, otherwise use zeros
                if frame_idx > 0:
                    features[frame_idx, joint_idx] = features[frame_idx-1, joint_idx]
                else:
                    features[frame_idx, joint_idx] = [0, 0, 0, 0, 0, 0]
    
    # Reshape to [timesteps, num_joints*6]
    num_frames, num_joints, dims = features.shape
    features_flat = features.reshape(num_frames, num_joints * dims)
    
    return features_flat


def normalize_motion_data(motion_data: np.ndarray) -> np.ndarray:
    """
    Normalize motion data with improved preprocessing for MotionBERT.
    
    Args:
        motion_data: Array of shape [timesteps, features]
        
    Returns:
        Normalized motion data with enhanced features
    """
    # Extract root joint positions (first 3 values)
    root_positions = motion_data[:, :3]
    
    # Create root-relative coordinates
    centered_data = motion_data.copy()
    for i in range(0, motion_data.shape[1], 3):
        centered_data[:, i:i+3] -= root_positions
    
    # Calculate velocity and acceleration
    velocity = np.zeros_like(centered_data)
    velocity[1:] = centered_data[1:] - centered_data[:-1]
    
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    # Scale features to similar ranges
    pos_scale = 1.0
    vel_scale = 0.5
    acc_scale = 0.25
    
    # Combine features with appropriate scaling
    enhanced_data = np.concatenate([
        centered_data * pos_scale,
        velocity * vel_scale,
        acceleration * acc_scale
    ], axis=1)
    
    # Global normalization
    max_abs_val = np.max(np.abs(enhanced_data))
    if max_abs_val > 0:
        enhanced_data /= max_abs_val
    
    return enhanced_data


def tokenize_sequence_pooling(motion_data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Improved sequence tokenization with motion-aware pooling.
    
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
    
    # Calculate motion energy for each frame
    motion_energy = np.sum(np.abs(motion_data), axis=1)
    max_energy = np.max(motion_energy)
    if max_energy > 0:
        motion_energy = motion_energy / max_energy
    else:
        motion_energy = np.ones_like(motion_energy) / original_length
    
    # Adaptive pooling based on motion energy
    for i in range(target_length):
        start_idx = int(i * original_length / target_length)
        end_idx = int((i + 1) * original_length / target_length)
        
        if start_idx == end_idx:
            tokenized[i] = motion_data[start_idx]
        else:
            # Weight frames by motion energy
            weights = motion_energy[start_idx:end_idx]
            weights = weights / (np.sum(weights) + 1e-8)  # Add small epsilon to avoid division by zero
            tokenized[i] = np.average(motion_data[start_idx:end_idx], weights=weights, axis=0)
    
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
    
    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return self.y.numpy()


def create_data_loaders(X, y, batch_size, val_size=0.2, random_state=42):
    """
    Split data into train/val sets and return DataLoaders.
    """
    # Convert labels to long tensor
    y = torch.LongTensor(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )
    
    train_dataset = EmotionMotionDataset(X_train, y_train)
    val_dataset = EmotionMotionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


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


class EnhancedLSTMClassifier(nn.Module):
    """
    Enhanced LSTM classifier with attention, residual connections, and layer normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, config: Dict):
        super(EnhancedLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection with layer normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"])
        )
        
        # Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=config["dropout_rate"] if num_layers > 1 else 0
        )
        
        # Layer normalization for LSTM output
        self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=config["dropout_rate"],
            batch_first=True
        )
        
        # Layer normalization for attention output
        self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(config["dropout_rate"]),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2)
            ) for _ in range(2)
        ])
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x.view(-1, x.size(-1))).view(batch_size, -1, self.hidden_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out)
        
        # Global context with skip connection
        context = torch.mean(attn_out + lstm_out, dim=1)
        
        # Residual connections
        for residual_block in self.residual_blocks:
            residual = residual_block(context)
            context = F.relu(context + residual)
        
        # Classification
        out = self.classifier(context)
        return out


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


def calculate_class_weights(dataset: EmotionMotionDataset) -> torch.Tensor:
    """
    Calculate weights inversely proportional to class frequencies.
    
    Args:
        dataset: EmotionMotionDataset instance
        
    Returns:
        Tensor of class weights for balanced training
    """
    y = dataset.get_labels()
    num_classes = len(EMOTION_MAP)  # Use exact number of classes from EMOTION_MAP
    
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


def train_motion_bert(model: nn.Module, 
                     train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader,
                     config: Dict) -> Tuple[nn.Module, Dict]:
    """Train MotionBERT model with enhanced training strategy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config["epochs"]
    num_warmup_steps = config["warmup_steps"]
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    best_model = None
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_val"])
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
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


def extract_joint_angles(positions: np.ndarray, joint_triplets: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Extract joint angles from position data.
    
    Args:
        positions: Array of shape [timesteps, num_joints*3]
        joint_triplets: List of (joint1, joint2, joint3) indices for angle calculation
        
    Returns:
        Array of joint angles
    """
    timesteps = positions.shape[0]
    num_angles = len(joint_triplets)
    angles = np.zeros((timesteps, num_angles))
    
    # Reshape positions to [timesteps, num_joints, 3]
    pos_reshaped = positions.reshape(timesteps, -1, 3)
    
    for i, (j1, j2, j3) in enumerate(joint_triplets):
        # Calculate vectors
        v1 = pos_reshaped[:, j1] - pos_reshaped[:, j2]
        v2 = pos_reshaped[:, j3] - pos_reshaped[:, j2]
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        v1_normalized = v1 / (v1_norm + 1e-7)
        v2_normalized = v2 / (v2_norm + 1e-7)
        
        # Calculate angles
        cos_angles = np.sum(v1_normalized * v2_normalized, axis=1)
        angles[:, i] = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    
    return angles


def enhance_motion_features(motion_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Extract enhanced motion features including velocities, accelerations, and joint angles.
    
    Args:
        motion_data: Array of shape [timesteps, num_joints*3]
        config: Configuration dictionary
        
    Returns:
        Enhanced feature array
    """
    features = [motion_data]  # Start with position data
    
    if config["use_velocity"]:
        velocity = extract_velocity_features(motion_data)
        features.append(velocity)
    
    if config["use_acceleration"]:
        acceleration = extract_acceleration_features(motion_data)
        features.append(acceleration)
    
    if config["use_joint_angles"]:
        # Define important joint angle triplets
        joint_triplets = [
            (IMPORTANT_JOINTS.index("Spine"), IMPORTANT_JOINTS.index("Neck"), IMPORTANT_JOINTS.index("Head")),
            (IMPORTANT_JOINTS.index("LeftArm"), IMPORTANT_JOINTS.index("LeftForeArm"), IMPORTANT_JOINTS.index("LeftHand")),
            (IMPORTANT_JOINTS.index("RightArm"), IMPORTANT_JOINTS.index("RightForeArm"), IMPORTANT_JOINTS.index("RightHand")),
            (IMPORTANT_JOINTS.index("LeftUpLeg"), IMPORTANT_JOINTS.index("LeftLeg"), IMPORTANT_JOINTS.index("LeftFoot")),
            (IMPORTANT_JOINTS.index("RightUpLeg"), IMPORTANT_JOINTS.index("RightLeg"), IMPORTANT_JOINTS.index("RightFoot"))
        ]
        angles = extract_joint_angles(motion_data, joint_triplets)
        features.append(angles)
    
    # Concatenate all features
    enhanced_features = np.concatenate(features, axis=1)
    
    # Apply normalization
    scaler = StandardScaler()
    enhanced_features = scaler.fit_transform(enhanced_features)
    
    return enhanced_features


def augment_motion_data(motion_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Apply various data augmentation techniques to motion data.
    
    Args:
        motion_data: Array of shape [timesteps, features]
        config: Configuration dictionary
        
    Returns:
        Augmented motion data
    """
    if np.random.random() > config["augment_prob"]:
        return motion_data
    
    augmented_data = motion_data.copy()
    timesteps, num_features = augmented_data.shape
    
    # Random temporal scaling (speed variation)
    if np.random.random() < 0.5:
        scale_factor = np.random.uniform(0.8, 1.2)
        new_timesteps = int(timesteps * scale_factor)
        new_timesteps = max(min(new_timesteps, int(timesteps * 1.2)), int(timesteps * 0.8))
        
        # Interpolate each feature independently
        new_data = np.zeros((new_timesteps, num_features))
        time_old = np.linspace(0, 1, timesteps)
        time_new = np.linspace(0, 1, new_timesteps)
        
        for i in range(num_features):
            new_data[:, i] = np.interp(time_new, time_old, augmented_data[:, i])
        
        augmented_data = new_data
    
    # Random rotation around vertical axis
    if np.random.random() < 0.5:
        angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
        rot_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        
        # Apply rotation to each joint position (assuming 3D coordinates)
        num_joints = num_features // 3
        positions = augmented_data.reshape(-1, num_joints, 3)
        positions = np.einsum('ij,klj->kli', rot_matrix, positions)
        augmented_data = positions.reshape(-1, num_features)
    
    # Random noise addition
    if np.random.random() < 0.3:
        noise_level = np.random.uniform(0.01, 0.02)
        noise = np.random.normal(0, noise_level, augmented_data.shape)
        augmented_data += noise
    
    # Random joint masking
    if np.random.random() < 0.3:
        num_joints = num_features // 3
        mask_size = np.random.randint(1, 4)  # Mask 1-3 joints
        joint_indices = np.random.choice(num_joints, mask_size, replace=False)
        for idx in joint_indices:
            start_idx = idx * 3
            augmented_data[:, start_idx:start_idx+3] = 0
    
    return augmented_data


def create_balanced_subset(X: np.ndarray, y: np.ndarray, samples_per_class: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a balanced subset of the dataset with specified number of samples per class.
    
    Args:
        X: Input features array
        y: Labels array
        samples_per_class: Number of samples to select per class
        
    Returns:
        Tuple of (X_subset, y_subset)
    """
    X_subset = []
    y_subset = []
    
    for class_label in np.unique(y):
        # Get indices for current class
        class_indices = np.where(y == class_label)[0]
        
        # Randomly select samples_per_class samples
        selected_indices = np.random.choice(class_indices, size=min(samples_per_class, len(class_indices)), replace=False)
        
        X_subset.extend(X[selected_indices])
        y_subset.extend(y[selected_indices])
    
    return np.array(X_subset), np.array(y_subset)


def process_dataset(data_dir: str, config: Dict) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Process the dataset and prepare it for training.
    
    Args:
        data_dir: Directory containing BVH files
        config: Configuration dictionary
        
    Returns:
        X: Motion features
        y: Labels
        metadata_df: DataFrame with metadata
    """
    # Get list of BVH files
    bvh_files = glob.glob(os.path.join(data_dir, "*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {data_dir}")
    
    print(f"Found {len(bvh_files)} BVH files")
    
    # Process each file
    X = []
    y = []
    metadata = []
    
    for bvh_file in tqdm(bvh_files, desc="Processing BVH files"):
        try:
            # Parse metadata from filename
            file_metadata = parse_filename(bvh_file)
            
            # Extract motion data
            motion_data = parse_bvh_file(bvh_file, config.get("selected_joints"))
            
            # Skip if motion data is invalid
            if motion_data is None or len(motion_data) < 10:
                continue
                
            # Enhance features
            if config["feature_engineering"]:
                motion_data = enhance_motion_features(motion_data, config)
            
            # Normalize sequence length
            motion_data = tokenize_sequence_pooling(motion_data, config["sequence_length"])
            
            X.append(motion_data)
            y.append(file_metadata["emotion_id"])
            metadata.append(file_metadata)
            
        except Exception as e:
            print(f"Error processing {bvh_file}: {str(e)}")
            continue
    
    if not X:
        raise ValueError("No valid motion data found in the dataset")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Print emotion distribution
    print("\nEmotion distribution:")
    for emotion_id, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {EMOTION_LABELS[emotion_id]}: {count} samples")
    
    # Data augmentation
    if config["use_data_augmentation"]:
        print("\nApplying data augmentation...")
        augmented_data = []
        augmented_labels = []
        
        # Compute class weights for balanced augmentation
        class_counts = np.bincount(y)
        max_count = np.max(class_counts)
        
        for class_id in range(len(class_counts)):
            class_indices = np.where(y == class_id)[0]
            num_augment = max_count - class_counts[class_id]
            
            if num_augment > 0:
                aug_indices = np.random.choice(class_indices, size=num_augment, replace=True)
                for idx in aug_indices:
                    aug_data = augment_motion_data(X[idx], config)
                    augmented_data.append(aug_data)
                    augmented_labels.append(class_id)
        
        if augmented_data:
            X = np.concatenate([X, np.array(augmented_data)], axis=0)
            y = np.concatenate([y, np.array(augmented_labels)])
    
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
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    if not os.path.exists(CONFIG["data_dir"]):
        print(f"Data directory {CONFIG['data_dir']} not found.")
        return
    
    print("Processing dataset...")
    X, y, metadata_df = process_dataset(CONFIG["data_dir"], CONFIG)
    
    # Save processed data
    print("Saving processed data...")
    np.save(os.path.join(CONFIG["output_dir"], "X.npy"), X)
    np.save(os.path.join(CONFIG["output_dir"], "y.npy"), y)
    metadata_df.to_csv(os.path.join(CONFIG["output_dir"], "metadata.csv"), index=False)
    
    # Split data
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
    
    # Create MotionBERT model
    print("Creating MotionBERT model...")
    model_config = create_motion_bert_config(
        input_dim=X_train.shape[2],  # features
        num_classes=len(EMOTION_MAP)
    )
    
    # Update model config with training parameters
    model_config.update({
        "learning_rate": CONFIG["learning_rate"],
        "weight_decay": CONFIG["weight_decay"],
        "epochs": CONFIG["epochs"],
        "early_stopping_patience": CONFIG["early_stopping_patience"],
        "warmup_steps": CONFIG["warmup_steps"],
        "max_steps": CONFIG["max_steps"],
        "gradient_clip_val": CONFIG["gradient_clip_val"]
    })
    
    model = MotionBERTClassifier(model_config)
    
    # Train model
    print("Training MotionBERT model...")
    model, history = train_motion_bert(model, train_loader, val_loader, model_config)
    
    # Plot training history
    plot_training_history(history, CONFIG)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "motion_bert_model.pth"))
    
    print("Done!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Emotion Recognition from Motion Data Pipeline')
    parser.add_argument('--data_dir', type=str, default='./test_data',
                        help='Directory containing BVH files')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    CONFIG["data_dir"] = args.data_dir
    CONFIG["output_dir"] = args.output_dir
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    
    main() 
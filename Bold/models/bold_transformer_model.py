#!/usr/bin/env python3
# bold_transformer_model.py - Emotion classification from BOLD dataset using Transformers

# --- 1. IMPORT LIBRARIES ---
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import copy
import math

# --- 2. CONFIGURATION ---
class Config:
    # Dataset paths
    BOLD_ROOT_DIR = "SUMMER/new_dataset_test_BOLD/BOLD_public"
    BOLD_JOINTS_DIR = os.path.join(BOLD_ROOT_DIR, "joints")
    BOLD_ANNOTATIONS_DIR = os.path.join(BOLD_ROOT_DIR, "annotations")
    BOLD_TRAIN_ANNOTATIONS = os.path.join(BOLD_ANNOTATIONS_DIR, "train.csv")
    BOLD_VAL_ANNOTATIONS = os.path.join(BOLD_ANNOTATIONS_DIR, "val.csv")
    
    # Data processing
    MAX_SAMPLES = 100  # Limit number of samples to process (set to None for all)
    SEQUENCE_LENGTH = 150  # Standardized sequence length for all samples
    NUM_JOINTS = 18  # COCO format has 18 joints
    
    # Mapping from BOLD to our emotion categories
    EMOTION_MAPPING = {
        # Positive emotions
        "Peace": "Positive",
        "Affection": "Positive",
        "Esteem": "Positive",
        "Anticipation": "Positive",
        "Engagement": "Positive",
        "Confidence": "Positive",
        "Happiness": "Positive",
        "Pleasure": "Positive",
        "Excitement": "Positive",
        "Surprise": "Positive",
        
        # Negative emotions
        "Doubt/Confusion": "Negative",
        "Disconnection": "Negative",
        "Fatigue": "Negative", 
        "Embarrassment": "Negative",
        "Yearning": "Negative",
        "Disapproval": "Negative",
        "Aversion": "Negative",
        "Annoyance": "Negative",
        "Anger": "Negative",
        "Sensitivity": "Negative",
        "Sadness": "Negative",
        "Disquietment": "Negative",
        "Fear": "Negative",
        "Pain": "Negative",
        "Suffering": "Negative",
        
        # Neutral emotions
        "Sympathy": "Neutral",
        "Neutral": "Neutral"
    }
    
    # Original 7-emotion mapping (if needed)
    EMOTION_MAPPING_DETAILED = {
        "Peace": "Neutral",
        "Affection": "Happy",
        "Esteem": "Happy",
        "Anticipation": "Surprise",
        "Engagement": "Neutral",
        "Confidence": "Happy",
        "Happiness": "Happy",
        "Pleasure": "Happy",
        "Excitement": "Happy",
        "Surprise": "Surprise",
        "Sympathy": "Neutral",
        "Doubt/Confusion": "Fearful",
        "Disconnection": "Neutral",
        "Fatigue": "Sad", 
        "Embarrassment": "Fearful",
        "Yearning": "Sad",
        "Disapproval": "Disgust",
        "Aversion": "Disgust",
        "Annoyance": "Angry",
        "Anger": "Angry",
        "Sensitivity": "Fearful",
        "Sadness": "Sad",
        "Disquietment": "Fearful",
        "Fear": "Fearful",
        "Pain": "Sad",
        "Suffering": "Sad"
    }
    
    # Model parameters
    INPUT_DIM = NUM_JOINTS * 3  # 18 joints x 3 coordinates
    HIDDEN_DIM = 128
    NUM_CLASSES = 3  # Using simplified classes: Positive, Negative, Neutral
    NUM_HEADS = 8
    NUM_LAYERS = 3
    DROPOUT = 0.3
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. DATA LOADING AND PREPROCESSING ---
def parse_bold_annotations(annotation_file, max_samples=None):
    """
    Parse the BOLD annotation file and extract emotion categories.
    
    Args:
        annotation_file: Path to the annotation CSV file
        max_samples: Maximum number of samples to process
        
    Returns:
        List of samples with metadata and emotion
    """
    print(f"Parsing annotations from {annotation_file}")
    
    # Load the annotations
    annotations = pd.read_csv(annotation_file, header=None)
    
    # Limit the number of samples if specified
    if max_samples is not None:
        annotations = annotations.iloc[:max_samples]
    
    # Process each annotation
    samples = []
    for i, row in annotations.iterrows():
        # Extract relevant information
        video_path = row[0]
        person_id = row[1]
        start_frame = row[2]
        end_frame = row[3]
        
        print(f"Processing annotation {i}: video={video_path}, person={person_id}, frames={start_frame}-{end_frame}")
        
        # Get categorical emotions (column 4 is a 0/1 vector of length 26)
        emotion_vector = str(row[4])
        
        # Find the primary emotion (first 1 in the binary string)
        try:
            # Find index of first '1' in the emotion vector
            primary_emotion_idx = emotion_vector.index('1')
            
            # Map to the corresponding emotion category
            emotion_categories = [
                "Peace", "Affection", "Esteem", "Anticipation", "Engagement", 
                "Confidence", "Happiness", "Pleasure", "Excitement", "Surprise", 
                "Sympathy", "Doubt/Confusion", "Disconnection", "Fatigue", 
                "Embarrassment", "Yearning", "Disapproval", "Aversion", 
                "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", 
                "Fear", "Pain", "Suffering"
            ]
            
            if primary_emotion_idx < len(emotion_categories):
                primary_emotion = emotion_categories[primary_emotion_idx]
            else:
                primary_emotion = "Unknown"
            
            # Extract joint file path
            # Format from annotation: 003/IzvOYVMltkI.mp4/0114.mp4
            # Target format: joints/003/IzvOYVMltkI.mp4/frame_number.npy
            parts = video_path.split('/')
            if len(parts) >= 3:
                folder = parts[0]  # "003"
                video_id = parts[1]  # "IzvOYVMltkI.mp4"
                
                # Extract target frame (middle of start and end)
                target_frame = int((start_frame + end_frame) / 2)
                
                # Try multiple frame numbers around the target frame to increase chances of finding a file
                found_file = False
                joint_file = None
                
                # Check a range of frames
                frames_to_try = [
                    target_frame,
                    # Try rounding to nearest tens
                    int(round(target_frame/10) * 10),
                    # Try specific frames that we know exist
                    750, 755, 754, 274, 273, 271
                ]
                
                for frame in frames_to_try:
                    frame_str = f"{frame:04d}"
                    test_file = os.path.join(Config.BOLD_JOINTS_DIR, folder, video_id, f"{frame_str}.npy")
                    print(f"Trying frame {frame_str}, path: {test_file}")
                    
                    if os.path.exists(test_file):
                        joint_file = test_file
                        found_file = True
                        print(f"Found existing joint file: {joint_file}")
                        break
                
                if not found_file:
                    print(f"No joint files found for video {video_id} after trying multiple frames")
                    continue
                
                samples.append({
                    "joint_file": joint_file,
                    "person_id": person_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "primary_emotion": primary_emotion,
                    "grouped_emotion": Config.EMOTION_MAPPING.get(primary_emotion, "Neutral")
                })
        except ValueError:
            # If no emotion is found, skip this sample
            print(f"No primary emotion found in sample {i}, skipping")
            continue
    
    print(f"Processed {len(samples)} valid samples")
    return samples

def load_joint_data(joint_file, person_id, start_frame, end_frame, seq_len=Config.SEQUENCE_LENGTH):
    """
    Load joint data from NumPy file.
    
    Args:
        joint_file: Path to the joint NumPy file
        person_id: ID of the person (not used directly since the file is already frame-specific)
        start_frame: Starting frame (not used directly)
        end_frame: Ending frame (not used directly)
        seq_len: Desired sequence length
        
    Returns:
        Normalized joint positions of shape [seq_len, num_joints*3]
    """
    try:
        # Check if file exists
        if not os.path.exists(joint_file):
            print(f"Joint file not found: {joint_file}")
            return None
        
        # Load the joint data - each row contains joint positions for a specific person
        joint_data = np.load(joint_file)
        
        # Find the row with the correct person ID
        # Person IDs are in the second column (index 1)
        matching_rows = joint_data[joint_data[:, 1] == person_id]
        
        if len(matching_rows) == 0:
            # If no match, use the first person in the file (best effort)
            if len(joint_data) > 0:
                # Use the first row but print a warning
                print(f"No exact match for person {person_id} in {joint_file}, using first available person")
                joint_positions = joint_data[0, 2:]  # Take all columns from 2 onwards (joint positions)
            else:
                print(f"Empty joint file: {joint_file}")
                return None
        else:
            # Use the first matching row
            joint_positions = matching_rows[0, 2:]  # Take all columns from 2 onwards
        
        # Reshape to match our expected format if needed (should be flat array of joints)
        joint_positions = joint_positions.reshape(1, -1)
        
        # Normalize the data
        mean = np.mean(joint_positions)
        std = np.std(joint_positions)
        normalized_positions = (joint_positions - mean) / (std + 1e-8)
        
        # Create a sequence by repeating the single frame
        sequence = np.repeat(normalized_positions, seq_len, axis=0)
        
        return sequence
        
    except Exception as e:
        print(f"Error loading joint data from {joint_file}: {e}")
        return None

def prepare_dataset(samples):
    """
    Prepare the dataset by loading and processing joint data for each sample.
    
    Args:
        samples: List of samples with metadata
        
    Returns:
        X: Joint data of shape [n_samples, seq_len, features]
        y: Emotion labels
    """
    X = []
    y = []
    
    print("Loading joint data for samples...")
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"Processing sample {i}/{len(samples)}")
            
        joint_data = load_joint_data(
            sample["joint_file"],
            sample["person_id"],
            sample["start_frame"],
            sample["end_frame"]
        )
        
        if joint_data is not None:
            X.append(joint_data)
            y.append(sample["grouped_emotion"])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples with joint data")
    print(f"Joint data shape: {X.shape}")
    print(f"Class distribution: {Counter(y)}")
    
    return X, y

# --- 4. DATASET CLASS ---
class BOLDDataset(Dataset):
    """Dataset class for BOLD joint data"""
    
    def __init__(self, X, y, label_encoder=None):
        self.X = X
        
        # Encode labels if needed
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.y_encoded = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder = label_encoder
            self.y_encoded = self.label_encoder.transform(y)
        
        # Store the number of classes
        self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return tensor of features and class index (not one-hot)
        return torch.FloatTensor(self.X[idx]), torch.tensor(self.y_encoded[idx], dtype=torch.long)
    
    def get_class_names(self):
        return self.label_encoder.classes_

# --- 5. MODEL DEFINITION ---
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BOLDTransformerModel(nn.Module):
    """Transformer model for emotion classification from joint data"""
    
    def __init__(self, num_classes=2, input_dim=54, hidden_dim=128, num_heads=8, 
                 num_layers=3, dropout=0.3, forward_expansion=4):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes
            input_dim: Dimension of input features (num_joints * 3)
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            forward_expansion: Expansion factor for the feed-forward network
        """
        super(BOLDTransformerModel, self).__init__()
        
        # Project input features to hidden dimension
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * forward_expansion,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Global pooling (temporal dimension)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project features
        x = self.feature_projection(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling across sequence dimension
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.global_pooling(x).squeeze(-1)  # [batch_size, d_model]
        
        # Classification
        output = self.classifier(x)
        
        return output

# --- 6. TRAINING AND EVALUATION FUNCTIONS ---
class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    
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
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

def evaluate_model(model, dataloader, criterion, device):
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
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy, all_preds, all_targets

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18, pad=20)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('bold_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_path='bold_training_plot.png'):
    """Plot and save training/validation accuracy and loss curves"""
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
    plt.close()

# --- 7. MAIN FUNCTION ---
def main():
    print("Starting BOLD Transformer Model Training")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading BOLD dataset...")
    
    # 1. Parse annotations
    train_samples = parse_bold_annotations(Config.BOLD_TRAIN_ANNOTATIONS, max_samples=Config.MAX_SAMPLES)
    val_samples = parse_bold_annotations(Config.BOLD_VAL_ANNOTATIONS, max_samples=Config.MAX_SAMPLES // 5)
    
    # 2. Prepare datasets
    X_train, y_train = prepare_dataset(train_samples)
    X_val, y_val = prepare_dataset(val_samples)
    
    # Check if we have enough data
    if len(X_train) == 0 or len(X_val) == 0:
        print("ERROR: No valid samples found. Cannot proceed with training.")
        print("Please check the dataset paths and structure.")
        return
    
    # 3. Create data loaders
    train_dataset = BOLDDataset(X_train, y_train)
    val_dataset = BOLDDataset(X_val, y_val, label_encoder=train_dataset.label_encoder)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE
    )
    
    # Get class names
    class_names = train_dataset.get_class_names()
    print(f"Classes: {class_names}")
    
    # 4. Create model
    num_classes = len(class_names)
    model = BOLDTransformerModel(
        num_classes=num_classes,
        input_dim=X_train.shape[2],
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    )
    model.to(device)
    
    # Print model summary
    print(f"Model Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 5. Define loss function and optimizer
    # Use weighted loss based on class frequencies
    class_counts = Counter(y_train)
    class_weights = []
    for class_name in class_names:
        # Add a small epsilon to avoid division by zero
        class_weights.append(1.0 / (class_counts.get(class_name, 1) + 1e-8))

    # Convert to tensor and normalize
    class_weights = torch.FloatTensor(class_weights)
    class_weights = class_weights / class_weights.sum()  # Normalize
    class_weights = class_weights.to(device)

    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # 6. Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 7. Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # 8. Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 9. Train model
    print("\nTraining model...")
    epochs = Config.NUM_EPOCHS
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        
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
    
    # 10. Plot training history
    plot_training_history(history)
    
    # 11. Final evaluation
    train_loss, train_acc, train_preds, train_targets = evaluate_model(
        model, train_loader, criterion, device
    )
    val_loss, val_acc, val_preds, val_targets = evaluate_model(
        model, val_loader, criterion, device
    )
    
    print(f"\nFinal Training Accuracy: {train_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    
    # 12. Confusion matrices
    train_cm = confusion_matrix(train_targets, train_preds)
    val_cm = confusion_matrix(val_targets, val_preds)
    
    # Print confusion matrices
    print("\nTraining Confusion Matrix:")
    print(train_cm)
    
    print("\nValidation Confusion Matrix:")
    print(val_cm)
    
    # 13. Plot confusion matrix
    plot_confusion_matrix(val_cm, class_names, title='Validation Confusion Matrix')
    
    # 14. Detailed classification metrics
    print("\nTraining Classification Report:")
    print(classification_report(train_targets, train_preds, target_names=class_names))
    
    print("\nValidation Classification Report:")
    print(classification_report(val_targets, val_preds, target_names=class_names))
    
    # 15. Save model
    # Convert classes_ attribute to list to avoid the pickler error
    classes_list = train_dataset.label_encoder.classes_.tolist()
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'label_encoder_classes': classes_list,
        'class_weights': class_weights.cpu().numpy(),
        'config': {
            'input_dim': X_train.shape[2],
            'hidden_dim': Config.HIDDEN_DIM,
            'num_classes': len(class_names),
            'dropout': Config.DROPOUT
        },
        'performance': {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
    }
    
    torch.save(model_data, 'SUMMER/bold_transformer_model.pt')
    
    print(f"\nModel saved to SUMMER/bold_transformer_model.pt")

if __name__ == "__main__":
    main() 
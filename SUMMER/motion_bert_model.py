#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MotionBERT-based Emotion Recognition Model

This module implements a MotionBERT-based model for emotion recognition from motion data.
MotionBERT is a transformer-based architecture specifically designed for motion understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class JointAttention(nn.Module):
    def __init__(self, d_model: int, num_joints: int):
        super().__init__()
        self.joint_attention = nn.MultiheadAttention(d_model, 8, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to [num_joints, batch * seq_len, features]
        batch_size, seq_len, num_joints, features = x.shape
        x = x.transpose(1, 2).reshape(num_joints, batch_size * seq_len, features)
        
        # Apply self-attention across joints
        attended, _ = self.joint_attention(x, x, x)
        attended = self.dropout(attended)
        x = self.norm(x + attended)
        
        # Reshape back
        x = x.reshape(num_joints, batch_size, seq_len, features)
        x = x.permute(1, 2, 0, 3)  # [batch, seq_len, num_joints, features]
        return x

class TemporalAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(d_model, 8, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, num_joints * features]
        x = x.transpose(0, 1)  # [seq_len, batch, features]
        attended, _ = self.temporal_attention(x, x, x)
        attended = self.dropout(attended)
        x = self.norm(x + attended)
        return x.transpose(0, 1)

class MotionBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection with improved feature extraction
        self.input_proj = nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.Dropout(config["dropout_rate"]),
            nn.GELU(),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.Dropout(config["dropout_rate"])
        )
        
        # Motion-specific positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config["hidden_dim"],
            dropout=config["dropout_rate"],
            max_len=config["max_sequence_length"]
        )
        
        # Transformer encoder with improved regularization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["hidden_dim"] * 4,
            dropout=config["dropout_rate"],
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"]
        )
        
        # Motion-specific attention
        self.motion_attention = nn.MultiheadAttention(
            embed_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            dropout=config["dropout_rate"],
            batch_first=True
        )
        
        # Improved temporal pooling
        self.temporal_pooling = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"])
        )
        
        # Classification head with improved regularization
        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
            nn.LayerNorm(config["hidden_dim"] // 2),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"] // 2, config["num_classes"])
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved defaults for motion data."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "classifier" in name:
                    # Initialize classification layers with smaller weights
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                else:
                    # Initialize other layers with standard weights
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        # Project input to hidden dimension for residual connection
        identity = self.input_proj[0](x)  # Only use first linear layer
        
        # Full input projection
        x = self.input_proj(x)
        x = x + identity  # Residual connection
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Motion-specific attention with residual connection
        attn_output, attn_weights = self.motion_attention(x, x, x)
        x = x + attn_output  # Residual connection
        
        # Temporal pooling with motion-aware mechanism
        # Use attention weights to weight the temporal pooling
        if attn_weights is not None:
            weights = attn_weights.mean(dim=1)  # Average across heads
            x = (x * weights.unsqueeze(-1)).sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        x = self.temporal_pooling(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

class MotionBERTClassifier(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        self.d_model = config["d_model"]
        self.num_joints = len(config["selected_joints"])
        input_dim = self.num_joints * 3  # 3D position per joint
        
        if config["use_velocity"]:
            input_dim += self.num_joints * 3
        if config["use_acceleration"]:
            input_dim += self.num_joints * 3
        if config["use_joint_angles"]:
            input_dim += 5  # Number of angle triplets
            
        # Input projection
        self.input_projection = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Joint attention
        self.joint_attention = JointAttention(self.d_model, self.num_joints)
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config["nhead"],
            dim_feedforward=config["hidden_dim"],
            dropout=config["dropout_rate"],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"]
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(self.d_model)
        
        # Output layers
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config["dropout_rate"])
        
        # Two-stage classification head
        self.pre_classifier = nn.Sequential(
            nn.Linear(self.d_model, config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2)
        )
        
        self.classifier = nn.Linear(config["hidden_dim"] // 2, config["num_classes"])
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Reshape for joint attention
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_joints, -1)
        
        # Apply joint attention
        x = self.joint_attention(x)
        
        # Reshape for temporal transformer
        x = x.reshape(batch_size, seq_len, -1)
        
        # Apply temporal attention
        x = self.temporal_attention(x)
        
        # Apply transformer encoder
        if attention_mask is not None:
            x = x.masked_fill(attention_mask.unsqueeze(-1), 0)
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.norm(x)
        x = self.dropout(x)
        x = self.pre_classifier(x)
        x = F.gelu(x)
        x = self.classifier(x)
        
        return x

def create_motion_bert_config(base_config: Dict) -> Dict:
    """Create MotionBERT specific configuration."""
    config = base_config.copy()
    
    # Add model specific parameters
    config.update({
        "max_position_embeddings": config["sequence_length"],
        "hidden_dropout_prob": config["dropout_rate"],
        "attention_probs_dropout_prob": config["dropout_rate"]
    })
    
    return config

def train_motion_bert(model, train_loader, val_loader, config):
    """Train MotionBERT model with improved training strategy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with improved settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps
    )
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_val"])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return model, history 
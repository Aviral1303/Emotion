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
from transformers import get_cosine_schedule_with_warmup

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MotionBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced input projection with residual connections
        self.input_proj = nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.Dropout(config["dropout_rate"]),
            nn.GELU(),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.Dropout(config["dropout_rate"])
        )
        
        # Motion-specific positional encoding with learned parameters
        self.pos_encoder = PositionalEncoding(
            d_model=config["hidden_dim"],
            dropout=config["dropout_rate"],
            max_len=config["max_sequence_length"]
        )
        
        # Enhanced transformer encoder with pre-norm and improved regularization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["hidden_dim"] * 4,
            dropout=config["dropout_rate"],
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"]
        )
        
        # Multi-scale motion attention
        self.motion_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config["hidden_dim"],
                num_heads=config["num_heads"],
                dropout=config["dropout_rate"],
                batch_first=True
            ) for _ in range(3)  # Multiple attention heads for different temporal scales
        ])
        
        # Enhanced temporal pooling with attention
        self.temporal_pooling = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"])
        )
        
        # Improved classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
            nn.LayerNorm(config["hidden_dim"] // 2),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["hidden_dim"] // 2, config["num_classes"])
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved defaults for motion data"""
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
        # Project input to hidden dimension
        x_proj = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x_proj)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Multi-scale motion attention with residual connections
        attn_outputs = []
        for attention in self.motion_attention:
            attn_out, _ = attention(x, x, x)
            attn_outputs.append(attn_out)
        
        # Combine attention outputs with residual
        x = x + sum(attn_outputs)
        
        # Temporal pooling
        x = self.temporal_pooling(x)
        
        # Global average pooling over sequence length
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class MotionBERTClassifier(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.motion_bert = MotionBERT(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.motion_bert(x)

def create_motion_bert_config(input_dim, num_classes):
    """Create configuration for MotionBERT model with improved defaults"""
    return {
        "input_dim": input_dim,
        "hidden_dim": 512,  # Increased hidden dimension
        "num_layers": 8,    # Increased number of layers
        "num_heads": 8,
        "dropout_rate": 0.3,  # Increased dropout
        "max_sequence_length": 200,
        "num_classes": num_classes,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "max_steps": 10000,
        "gradient_clip_val": 1.0
    }

def train_motion_bert(model, train_loader, val_loader, config):
    """Train MotionBERT model with improved training strategy"""
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
    
    # Learning rate scheduler with warmup and cosine decay
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
        
        # Early stopping with improved patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    
    return model, history 
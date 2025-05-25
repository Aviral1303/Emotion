#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Experiment Runner for Emotion Recognition

This script runs experiments with improved data processing and model architecture
to achieve higher accuracy in emotion recognition.
"""

import os
import subprocess
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import argparse

def run_experiment(samples_per_emotion):
    """Run a single experiment with specified number of samples per emotion."""
    # Create experiment directory
    exp_dir = f"experiment_{samples_per_emotion}_samples"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Extract test dataset
    subprocess.run([
        "python3", "extract_test_dataset.py",
        "--samples", str(samples_per_emotion)
    ])
    
    # Enhanced configuration
    config = {
        "data_dir": "./test_data",
        "output_dir": exp_dir,
        "sequence_length": 300,
        "batch_size": 16,
        "epochs": 150,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "dropout_rate": 0.2,
        "use_data_augmentation": True,
        "feature_engineering": True,
        "use_velocity": True,
        "use_acceleration": True,
        "use_joint_angles": True,
        "use_selected_joints": True,
        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,
        "hidden_dim": 256,
        "num_classes": 7,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "augment_prob": 0.5,
        "use_focal_loss": True,
        "focal_gamma": 2.0,
        "warmup_steps": 2000,
        "max_steps": 20000
    }
    
    # Save configuration
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Run emotion recognition pipeline
    subprocess.run([
        "python3", "emotion_recognition_pipeline.py",
        "--config", os.path.join(exp_dir, "config.json")
    ])
    
    # Load results
    try:
        with open(os.path.join(exp_dir, "results.json"), "r") as f:
            results = json.load(f)
        return {
            "samples_per_emotion": samples_per_emotion,
            "train_acc": results["train_acc"],
            "val_acc": results["val_acc"],
            "test_acc": results["test_acc"],
            "best_epoch": results["best_epoch"]
        }
    except Exception as e:
        print(f"Error loading results for {samples_per_emotion} samples: {str(e)}")
        return None

def plot_results(results_df):
    """Plot experiment results with enhanced visualization."""
    plt.figure(figsize=(15, 10))
    
    # Plot training metrics
    plt.subplot(2, 1, 1)
    plt.plot(results_df["samples_per_emotion"], results_df["train_acc"],
             marker='o', label='Training Accuracy', linewidth=2)
    plt.plot(results_df["samples_per_emotion"], results_df["val_acc"],
             marker='s', label='Validation Accuracy', linewidth=2)
    plt.plot(results_df["samples_per_emotion"], results_df["test_acc"],
             marker='^', label='Test Accuracy', linewidth=2)
    plt.xlabel('Samples per Emotion Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance vs Dataset Size')
    plt.legend()
    plt.grid(True)
    
    # Plot best epoch distribution
    plt.subplot(2, 1, 2)
    plt.bar(results_df["samples_per_emotion"], results_df["best_epoch"])
    plt.xlabel('Samples per Emotion Class')
    plt.ylabel('Best Epoch')
    plt.title('Convergence Speed vs Dataset Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("experiment_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run emotion recognition experiments')
    parser.add_argument('--samples', type=int, default=20,
                      help='Number of samples per emotion class')
    parser.add_argument('--epochs', type=int, default=150,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    args = parser.parse_args()
    
    # Sample sizes to test
    base_samples = args.samples
    sample_sizes = [
        max(5, base_samples // 4),
        max(10, base_samples // 2),
        max(15, base_samples * 3 // 4),
        base_samples
    ]
    
    # Run experiments in parallel
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_experiment, sample_sizes))
    
    # Filter out failed experiments
    results = [r for r in results if r is not None]
    
    if not results:
        print("No experiments completed successfully")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv("experiment_results.csv", index=False)
    
    # Plot results
    plot_results(results_df)
    
    # Print summary
    print("\nExperiment Results Summary:")
    print(results_df.to_string(index=False))
    
    # Find best configuration
    best_config = results_df.loc[results_df["test_acc"].idxmax()]
    
    print("\nBest Configuration:")
    print(f"Samples per emotion: {best_config['samples_per_emotion']}")
    print(f"Training Accuracy: {best_config['train_acc']:.2f}%")
    print(f"Validation Accuracy: {best_config['val_acc']:.2f}%")
    print(f"Test Accuracy: {best_config['test_acc']:.2f}%")
    print(f"Best Epoch: {best_config['best_epoch']}")

if __name__ == "__main__":
    main() 
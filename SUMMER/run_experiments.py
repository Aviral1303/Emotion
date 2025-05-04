#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Experiment Runner for Emotion Recognition

This script runs multiple experiments with different sample sizes in parallel
to find the optimal configuration for emotion recognition.
"""

import os
import subprocess
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def run_experiment(samples_per_emotion):
    """Run a single experiment with specified number of samples per emotion."""
    # Create experiment directory
    exp_dir = f"experiment_{samples_per_emotion}_samples"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Extract test dataset
    subprocess.run(["python3", "extract_test_dataset.py", "--samples", str(samples_per_emotion)])
    
    # Run emotion recognition pipeline
    subprocess.run([
        "python3", "emotion_recognition_pipeline.py",
        "--data_dir", "./test_data",
        "--output_dir", exp_dir,
        "--epochs", "20",
        "--batch_size", "4"
    ])
    
    # Load results
    try:
        with open(os.path.join(exp_dir, "results.json"), "r") as f:
            results = json.load(f)
        return {
            "samples_per_emotion": samples_per_emotion,
            "mlp_train_acc": results["mlp_train_acc"],
            "mlp_val_acc": results["mlp_val_acc"],
            "lstm_train_acc": results["lstm_train_acc"],
            "lstm_val_acc": results["lstm_val_acc"]
        }
    except:
        return None

def plot_results(results_df):
    """Plot experiment results."""
    plt.figure(figsize=(12, 6))
    
    # Plot MLP results
    plt.subplot(1, 2, 1)
    plt.plot(results_df["samples_per_emotion"], results_df["mlp_train_acc"], 
             marker='o', label='MLP Train')
    plt.plot(results_df["samples_per_emotion"], results_df["mlp_val_acc"], 
             marker='o', label='MLP Val')
    plt.xlabel('Samples per Emotion')
    plt.ylabel('Accuracy (%)')
    plt.title('MLP Model Performance')
    plt.legend()
    plt.grid(True)
    
    # Plot LSTM results
    plt.subplot(1, 2, 2)
    plt.plot(results_df["samples_per_emotion"], results_df["lstm_train_acc"], 
             marker='o', label='LSTM Train')
    plt.plot(results_df["samples_per_emotion"], results_df["lstm_val_acc"], 
             marker='o', label='LSTM Val')
    plt.xlabel('Samples per Emotion')
    plt.ylabel('Accuracy (%)')
    plt.title('LSTM Model Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("experiment_results.png")
    plt.close()

def main():
    # Sample sizes to test
    sample_sizes = [5, 10, 15, 20]
    
    # Run experiments in parallel
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_experiment, sample_sizes))
    
    # Filter out failed experiments
    results = [r for r in results if r is not None]
    
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
    best_mlp = results_df.loc[results_df["mlp_val_acc"].idxmax()]
    best_lstm = results_df.loc[results_df["lstm_val_acc"].idxmax()]
    
    print("\nBest Configurations:")
    print(f"MLP: {best_mlp['samples_per_emotion']} samples per emotion "
          f"(Val Acc: {best_mlp['mlp_val_acc']:.2f}%)")
    print(f"LSTM: {best_lstm['samples_per_emotion']} samples per emotion "
          f"(Val Acc: {best_lstm['lstm_val_acc']:.2f}%)")

if __name__ == "__main__":
    main() 
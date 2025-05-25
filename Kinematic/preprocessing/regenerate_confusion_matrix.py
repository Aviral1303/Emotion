#!/usr/bin/env python3
# Script to regenerate confusion matrices with improved visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_enhanced_confusion_matrix(confusion_matrix, class_names, title='Confusion Matrix', 
                                   save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix using seaborn heatmap with improved visualization.
    
    Args:
        confusion_matrix: The confusion matrix to visualize
        class_names: List of class names
        title: Title for the plot
        save_path: Path to save the plot
    """
    # Create a DataFrame for better visualization
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12, 10))
    
    # Calculate percentages for each row (true labels)
    row_sums = confusion_matrix.sum(axis=1)
    cm_norm = np.zeros_like(confusion_matrix, dtype=float)
    for i in range(len(confusion_matrix)):
        if row_sums[i] > 0:
            cm_norm[i] = confusion_matrix[i] / row_sums[i]
    
    # Format annotations to show both count and percentage
    annot = np.empty_like(confusion_matrix, dtype=object)
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            if row_sums[i] > 0:
                annot[i, j] = f"{confusion_matrix[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            else:
                annot[i, j] = f"{confusion_matrix[i, j]}\n(0.0%)"
    
    # Create heatmap with improved color scheme and annotations
    ax = sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', 
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 14}, vmin=0, vmax=1)
    
    # Set labels with increased font size
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=16, fontweight='bold', labelpad=15)
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=15)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add grid lines for better visualization
    for i in range(confusion_matrix.shape[0] + 1):
        plt.axhline(y=i, color='white', lw=2)
        plt.axvline(x=i, color='white', lw=2)
    
    # Add descriptive text
    total_samples = np.sum(row_sums)
    class_distribution = ", ".join([f"{class_names[i]}: {row_sums[i]} ({row_sums[i]/total_samples*100:.1f}%)" 
                                   for i in range(len(class_names))])
    
    plt.figtext(0.5, 0.01, f"Total samples: {total_samples}\nClass distribution: {class_distribution}",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced confusion matrix saved to {save_path}")
    plt.close()

def main():
    # Define the confusion matrices from the last run
    
    # Training Confusion Matrix
    train_cm = np.array([
        [92, 0, 3],    # Negative
        [18, 0, 1],    # Neutral
        [37, 0, 9]     # Positive
    ])
    
    # Test Confusion Matrix
    test_cm = np.array([
        [22, 0, 2],    # Negative
        [5, 0, 0],     # Neutral
        [8, 0, 3]      # Positive
    ])
    
    class_names = ['Negative', 'Neutral', 'Positive']
    
    # Generate enhanced confusion matrices
    plot_enhanced_confusion_matrix(
        train_cm, 
        class_names, 
        title='Training Set Confusion Matrix (200 samples)', 
        save_path='enhanced_training_confusion_matrix.png'
    )
    
    plot_enhanced_confusion_matrix(
        test_cm, 
        class_names, 
        title='Test Set Confusion Matrix (40 samples)', 
        save_path='enhanced_test_confusion_matrix.png'
    )

if __name__ == "__main__":
    main() 
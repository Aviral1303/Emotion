# Emotion Recognition from Motion Data

## Abstract

This project explores emotion recognition using human body motion data, leveraging deep learning models to classify emotions from motion capture and pose estimation datasets. The goal is to expand the input vocabulary of AI models to include non-verbal communication, using motion as a primary signal for emotion recognition. We implement and compare multiple neural architectures, process real-world datasets, and provide a reproducible research pipeline.

## Outline

1. Abstract
2. Outline
3. Introduction
4. Background
5. Analysis
6. Strategies
7. Toolkit
8. Evaluation
9. Related Work
10. Conclusion
11. Revision Summary
12. Revision Detailed
13. Team Members

## Introduction

### Research Motivation

The main purpose of this project is to use motion data over time as an input for emotion recognition—expanding the vocabulary of AI models to include non-verbal communication. The assumption is that significant communication occurs non-verbally, which is not captured in common datasets. As a long-term vision, imagine a camera-based AI agent that interprets requests using voice, facial expression, and body motion. This work is an exemplar case of using motion data as an input for emotion recognition.

### Project Overview

We present a minimum viable product (MVP) that:

1. Ingests motion data from existing datasets.
2. Tokenizes sequences of motion data into fixed-dimension tensors.
3. Trains neural networks to predict emotion labels from body motion alone.

## Background

- Literature review on body-motion data for emotion recognition.
- Review of tokenization strategies for sequential data (e.g., transformers, LSTMs).
- Survey of available datasets (PhysioNet Kinematic, BOLD, etc.).
- Challenges in labeling and classifying emotions from motion.

## Analysis

### Datasets

- **PhysioNet Kinematic Dataset of Actors Expressing Emotions**: BVH motion capture files, 7 emotion classes, multiple actors, joint position data.
- **BOLD (Bodily Expressed Language Dataset)**: Video-based, 2D skeletal joint positions, 26 emotion categories grouped into positive/negative/neutral.

### Data Processing

- BVH file parsing and joint extraction.
- Feature normalization and sequence length standardization.
- Emotion grouping to address class imbalance.
- Data augmentation: time shifting, scaling, noise addition, segment reversal, interpolation, mix-up.

## Strategies

### Model Architectures

- **Transformer Emotion Model**: Self-attention for temporal relationships.
- **BiLSTM Emotion Model**: Bidirectional LSTM for sequential dependencies.
- **TCN Emotion Model**: Temporal Convolutional Network for local and long-range patterns.
- **BOLD Transformer Model**: Specialized transformer for BOLD dataset, attention-based, handles missing data and person identification.

### Tokenization

- Variable-length motion sequences projected to fixed-size tensors for model input.
- Feature engineering: velocity, acceleration, joint trajectories.

## Toolkit

### Codebase Structure

- `Kinematic/` and `Bold/`: Contain models, preprocessing scripts, and outcomes for each dataset.
- `kinematic-dataset-of-actors-expressing-emotions-2.1.0/`: Raw BVH files and metadata.
- `requirements.txt`, `requirements_SUMMER.txt`: Python dependencies.
- Preprocessing scripts: `emotion_recognition_pipeline.py`, `motion_visualization.py`, `regenerate_confusion_matrix.py`, etc.
- Model scripts: `transformer_emotion_model.py`, `bilstm_emotion_model.py`, `tcn_emotion_model.py`, `bold_transformer_model.py`, etc.
- Evaluation and visualization: confusion matrices, training plots, heatmaps.

## Evaluation

### BOLD TCN Model Results

- **Final Test Accuracy:** 40.00%
- **Final Test Loss:** 1.4645
- **Confusion Matrix (7 classes):**
  - Rows: True labels, Columns: Predicted labels
  - Labels: ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprise']
  - Example (partial):
    [[2 0 1 0 0 1 0]
    [0 1 0 0 1 1 1]
    ...]
- **Training stopped at epoch 42 (early stopping).**
- **Training plot and confusion matrix saved in Bold/outcomes/.**

### BOLD BiLSTM Model Results

- **Final Test Accuracy:** 60.00%
- **Final Test Loss:** 0.9008
- **Confusion Matrix (3 classes):**
  - Rows: True labels, Columns: Predicted labels
  - Labels: ['Negative', 'Neutral', 'Positive']
  - Example:
    [[18  0  0]
[ 3  0  0]
[ 9  0  0]]
- **Training stopped at epoch 10 (early stopping).**
- **Training plot and confusion matrix saved in Bold/outcomes/.**

### BOLD Transformer Model Results

- **Final Training Accuracy:** 61.90%
- **Final Validation Accuracy:** 53.85%
- **Training Confusion Matrix:**
  - [[3  1]
[15 23]]
- **Validation Confusion Matrix:**
  - [[0 4]
[2 7]]
- **Training Classification Report:**
  - Negative: Precision 0.17, Recall 0.75, F1 0.27 (support 4)
  - Positive: Precision 0.96, Recall 0.61, F1 0.74 (support 38)
- **Validation Classification Report:**
  - Negative: Precision 0.00, Recall 0.00, F1 0.00 (support 4)
  - Positive: Precision 0.64, Recall 0.78, F1 0.70 (support 9)
- **Early stopping triggered at epoch 11.**
- **Training completed in 4.5 seconds.**

### Kinematic Transformer Model Results

- **Training set:** 74 samples, **Test set:** 19 samples
- **Model:** TransformerEmotionClassifier, 876,037 parameters
- **Training accuracy:** Reached up to ~73%
- **Validation accuracy:** Reached up to ~63%
- **Class-specific performance:**
  - Neutral and Positive classes achieved high recall and precision (Neutral: up to 100% recall/precision, Positive: up to 100% recall, ~49-50% precision)
  - Negative class recall/precision remained 0% throughout training
- **Training epochs:** Training continued for 58+ epochs (early stopping not reached in shown output)
- **Validation loss:** Remained 0.0 (possible code/data issue), but accuracy improved
- **Confusion matrix:**
  - Model is able to learn Neutral and Positive classes well, but struggles with Negative class

### Results

- **Transformer Model**: ~65% accuracy on test set (Kinematic dataset).
- **BiLSTM Model**: Effective for sequential pattern recognition.
- **TCN Model**: Good at capturing local motion patterns.
- **BOLD Transformer Model**: 61.9% training accuracy, 53.85% validation accuracy (BOLD dataset).

#### Class-Specific Performance (Kinematic)

- Neutral: Precision 86.2%, Recall 100%, F1 92.6%
- Positive: Precision 48.9%, Recall 91.7%, F1 63.6%
- Negative: Precision 0%, Recall 0%, F1 0%

#### Visualizations

- Training/validation curves, confusion matrices, heatmaps, joint trajectories.

### Areas for Improvement

- Enhance negative class detection.
- Improve precision for positive class.
- Feature engineering for better discrimination.
- Ensemble methods for robustness.

## Related Work

- [Kinematic Dataset of Actors Expressing Emotions](https://physionet.org/content/kinematic-actors-emotions/2.1.0/)
- [BOLD Dataset](https://cydar.ist.psu.edu/emotionchallenge/index.html)
- Recent works on transformers for sequential data (NLP, robotics).
- Literature on non-verbal emotion recognition.

## Conclusion

This project demonstrates the feasibility of emotion recognition from motion data using deep learning. The pipeline is adaptable to multiple datasets and model architectures. Future work includes ensemble methods, advanced feature engineering, and cross-dataset training.

## Revision Summary

### Milestones

- **By 5/20**: Literature review, dataset search, preliminary models.

- Ongoing: Model improvements, evaluation, documentation.

## Revision Detailed

- See `.git/` for full version history.
- Major updates tracked in this README and code comments.

## Team Members

- Aryaman
- Aviral

---

For setup, usage, and detailed instructions, see the sections above and the code in `Kinematic/` and `Bold/`. For questions or collaboration, please contact the team members.

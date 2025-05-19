# Emotion Recognition from Motion Capture Data

A comprehensive framework for recognizing emotions from human motion data using various deep learning architectures. This project analyzes motion capture (BVH) files from the PhysioNet Kinematic Dataset of Actors Expressing Emotions, implementing and comparing multiple model architectures.

## Project Overview

This project applies deep learning techniques to recognize human emotions from motion capture data. By analyzing joint movement patterns over time, our models can classify emotions such as happiness, anger, sadness, and more.

## Dataset

The project uses the [PhysioNet Kinematic Dataset of Actors Expressing Emotions](https://physionet.org/content/kinematic-actors-emotions/2.1.0/), which contains motion capture recordings of professional actors expressing various emotions through movement.

Key features:

- BVH (Biovision Hierarchy) motion capture files
- 7 emotion classes: Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprise
- Multiple actors performing various movement scenarios
- Joint position data for up to 20 key joints

### Data Splitting

For model training and evaluation, the dataset is split as follows:

- **Train/Test Split**: 80% training, 20% testing across all models
- **Validation**: 10% of training data used for validation during training (via validation_split parameter)
- **Stratification**: Stratified sampling ensures class distribution is maintained across splits
- **Cross-Actor**: Data from all actors is mixed in both training and testing sets
- **Subject Limitation**: For quick experiments, the first 100-150 files are used (configurable)
- **Emotion Grouping**: Optional grouping of 7 emotions into 3 broader categories to address class imbalance:
  - Positive: Happy, Surprise
  - Negative: Angry, Fearful, Sad, Disgust
  - Neutral: Neutral

## Models Implemented

### 1. Transformer Emotion Model

`transformer_emotion_model.py` implements a transformer-based architecture for emotion classification:

- **Architecture**: Uses self-attention mechanisms to capture temporal relationships in motion sequences
- **Input**: Normalized 3D joint positions (60 features × 150 time steps)
- **Key components**:
  - Dense embedding layer with dropout (0.3)
  - Positional encoding
  - Transformer encoder block with multi-head attention
  - Global average pooling
  - Classification head
- **Training**:
  - AdamW optimizer with weight decay (1e-4)
  - Early stopping with patience=5
  - Emotion grouping into 3 categories (Positive, Negative, Neutral)
- **Performance**: Achieved ~65% accuracy on the test set

### 2. BiLSTM Emotion Model

`bilstm_emotion_model.py` implements a bidirectional LSTM architecture:

- **Architecture**: Captures temporal patterns in both forward and backward directions
- **Input**: Normalized 3D joint positions (60 features × 150 time steps)
- **Key components**:
  - Bidirectional LSTM layers
  - Dropout regularization (0.3)
  - Linear classification head
- **Training**:
  - Adam optimizer
  - Early stopping
  - Class balancing through stratified sampling
- **Performance**: Effective at capturing motion sequences with temporal dependencies

### 3. TCN Emotion Model

`tcn_emotion_model.py` implements a Temporal Convolutional Network:

- **Architecture**: Uses dilated causal convolutions to maintain temporal causality
- **Input**: Normalized 3D joint positions (60 features × 150 time steps)
- **Key components**:
  - Multiple TCN blocks with increasing dilation
  - Residual connections
  - Dropout regularization (0.3)
  - Global average pooling
  - Classification head
- **Training**:
  - Adam optimizer
  - Early stopping
  - Learning rate scheduling
- **Performance**: Good at capturing local motion patterns with long effective receptive fields

## Key Features

- **Data Preprocessing**:

  - BVH file parsing and joint extraction
  - Feature normalization (mean subtraction and division by standard deviation)
  - Sequence length standardization
  - Optional emotion grouping into broader categories

- **Training Optimizations**:

  - Early stopping to prevent overfitting
  - Dropout regularization
  - L2 weight decay
  - Stratified sampling for balanced class distribution

- **Visualization**:
  - Training and validation metrics plotting
  - Confusion matrix analysis
  - Loss and accuracy curves

## Usage

1. Ensure the dataset is downloaded to the `kinematic-dataset-of-actors-expressing-emotions-2.1.0` directory
2. Install requirements: `pip install -r requirements.txt`
3. Run a specific model:
   ```
   python transformer_emotion_model.py
   python SUMMER/bilstm_emotion_model.py
   python SUMMER/tcn_emotion_model.py
   ```

## Results

The models were evaluated on their ability to classify emotions from motion data. Performance varies by architecture:

- **Transformer**: Good at capturing global dependencies in motion sequences
- **BiLSTM**: Effective for sequential pattern recognition
- **TCN**: Strong at identifying localized motion patterns with different temporal scales

## Future Work

- Ensemble methods combining multiple architectures
- Hyperparameter optimization
- Additional feature engineering beyond raw joint positions
- Cross-validation for more robust evaluation

## Latest Model Performance (Updated)

### Overall Performance
- Training Accuracy: 63.51%
- Test Accuracy: 63.16%

### Class-Specific Performance

#### Neutral Class
- Precision: 0.8621 (86.21%)
- Recall: 1.0000 (100.00%)
- F1-Score: 0.9260 (92.60%)
- Support: 31 samples

#### Positive Class
- Precision: 0.4889 (48.89%)
- Recall: 0.9167 (91.67%)
- F1-Score: 0.6364 (63.64%)
- Support: 31 samples

#### Negative Class
- Precision: 0.0000 (0.00%)
- Recall: 0.0000 (0.00%)
- F1-Score: 0.0000 (0.00%)
- Support: 31 samples

### Key Observations
1. The model shows excellent performance for the Neutral class with perfect recall and high precision
2. Good recall for Positive class (91.67%) but lower precision (48.89%)
3. The model currently struggles with Negative class detection
4. Class balancing was successful with 31 samples per class after augmentation

### Model Architecture
- Total Parameters: 876,037
- Input Features: 60 (20 joints × 3 coordinates)
- Sequence Length: 150 frames
- Number of Classes: 3 (Negative, Neutral, Positive)

### Training Details
- Optimizer: AdamW with learning rate 0.0003
- Loss Function: CrossEntropyLoss with custom class weights [0.2, 0.4, 0.6]
- Early Stopping: Patience of 15 epochs
- Learning Rate Scheduler: ReduceLROnPlateau with factor 0.5 and patience 5

### Data Augmentation
- Successfully balanced classes from original distribution:
  - Negative: 62.0%
  - Neutral: 12.0%
  - Positive: 26.0%
- Final balanced distribution: 33.3% per class
- Augmentation techniques used:
  - Time shifting
  - Time scaling
  - Noise addition
  - Segment reversal
  - Interpolation
  - Mix-up

### Areas for Improvement
1. Enhance Negative class detection
2. Improve precision for Positive class
3. Investigate feature engineering for better emotion discrimination
4. Consider ensemble methods for improved robustness

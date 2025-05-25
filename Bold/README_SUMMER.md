# Emotion Recognition from Motion Data

This research pipeline explores emotion recognition using body motion data. It processes BVH motion capture files, extracts features, and trains machine learning models to predict emotions from movement patterns.

## Dataset

This project uses the [Kinematic Dataset of Actors Expressing Emotions](https://physionet.org/content/kinematic-actors-emotions/1.0.0/), which contains motion capture recordings of actors expressing different emotions through their body movements.

The dataset contains BVH (Biovision Hierarchy) files with naming convention:

- `F01A0V1.bvh` = Female Actor 01, Angry emotion, Scenario 0, Version 1
- `M05H2V3.bvh` = Male Actor 05, Happy emotion, Scenario 2, Version 3

The emotion codes in filenames are:

- A: Angry
- D: Disgust
- F: Fearful
- H: Happy
- N: Neutral
- SA: Sad
- SU: Surprise

## Setup and Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd emotion-recognition-motion
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset from [PhysioNet](https://physionet.org/content/kinematic-actors-emotions/1.0.0/) and extract it to a `data` directory in the project root.

## Pipeline Components

This project consists of several components:

### 1. Data Processing (`emotion_recognition_pipeline.py`)

- **BVH File Parsing**: Extracts 3D joint positions from BVH files
- **Motion Normalization**: Centers and scales the motion data
- **Motion Tokenization**: Converts variable-length sequences to fixed-size inputs
- **Model Training**: Trains and evaluates emotion classification models

### 2. Visualization Tools (`motion_visualization.py`)

- **Skeleton Animation**: Creates 3D animations of motion sequences
- **Motion Heatmaps**: Visualizes motion patterns as heatmaps
- **Joint Trajectories**: Plots the trajectory of joints over time
- **Emotion Comparison**: Compares motion patterns across different emotions

## Usage Instructions

### Data Processing and Model Training

Run the main pipeline with default settings:

```bash
python emotion_recognition_pipeline.py
```

This will:

1. Create necessary directories (data, output)
2. Load and process BVH files from the data directory
3. Tokenize motion sequences to fixed length
4. Train MLP and LSTM models for emotion classification
5. Save processed data and trained models

### Visualization Tools

The visualization script provides several ways to explore the motion data:

1. **Visualize a specific motion file**:

```bash
python motion_visualization.py --file Actor_01_Happy_001.bvh
```

2. **Create a skeleton animation**:

```bash
python motion_visualization.py --file Actor_01_Happy_001.bvh --animate
```

3. **Compare emotions for a specific actor**:

```bash
python motion_visualization.py --compare --actor_id 1
```

4. **Customize output directory**:

```bash
python motion_visualization.py --file Actor_01_Happy_001.bvh --output_dir ./my_visualizations
```

## Research Pipeline Details

### Data Representation

Motion data is represented in the following ways:

- Raw format: `[timesteps, num_joints*3]` (3D positions of each joint)
- Tokenized format: `[100, features]` (fixed-length sequence for model input)

### Model Architectures

The pipeline supports multiple model architectures:

1. **MLP Classifier**: Simple feedforward network for baseline results
2. **LSTM Classifier**: Recurrent model that captures temporal patterns
3. **Transformer Classifier**: Advanced architecture based on attention mechanisms

### Output Files

The pipeline generates several output files:

- `X.npy`: Tokenized motion inputs
- `y.npy`: Corresponding emotion labels
- `metadata.csv`: File-level information
- `best_model.pt`: Best performing model weights
- `training_history.png`: Training and validation curves

## Extending the Pipeline

Here are some ways to extend this research pipeline:

1. **Custom Tokenization**: Implement more advanced tokenization methods in `emotion_recognition_pipeline.py`
2. **Feature Engineering**: Extract higher-level motion features (velocity, acceleration, etc.)
3. **Model Architecture**: Implement new model architectures in the `models.py` module
4. **Cross-Validation**: Add k-fold cross-validation for more robust evaluation
5. **Hyperparameter Tuning**: Implement grid search or random search for hyperparameter optimization

## License

This project is released under the MIT License.

## Acknowledgments

- [Kinematic Dataset of Actors Expressing Emotions](https://physionet.org/content/kinematic-actors-emotions/1.0.0/)
- Motion Capture Parser library

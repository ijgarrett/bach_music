# Bach Chorale Music Generation

This project uses a dataset of Bach chorales to train a neural network that can generate polyphonic music sequences. I implemented this project in September 2025, using convolutional and recurrent layers in TensorFlow/Keras to model temporal dependencies between notes. The trained model can generate new chorales based on a small seed sequence of chords.

## Project Overview

The goal is to generate sequences of notes (chorales) in the style of Bach. The model is trained on numerical representations of chorales and predicts the next note for each voice given a sequence of previous notes. Generated sequences can be converted back into MIDI format for playback.

## Dataset

- Contains MIDI-extracted Bach chorales converted to CSV
- Each CSV row represents one timestep, each column represents one voice
- Notes are encoded as integers:
  - 36 = C1, 81 = A5
  - 0 = silence
- Split into training, validation, and test sets

## Tools and Libraries

- Python  
- TensorFlow/Keras (`Sequential`, `Conv1D`, `Embedding`, `LSTM`, `Dense`, `Dropout`, `BatchNormalization`)  
- pandas & NumPy (data handling)  
- music21 (MIDI creation and playback)  
- Google Colab (GPU acceleration)

## Process and Methodology

### 1. Data Preprocessing
- Loaded chorale CSVs into Python lists
- Defined `window_size` and `window_offset` to create sequences for training
- Flattened sequences for model input
- Rescaled note integers to match embedding input

### 2. Model Architecture
- Embedding layer: maps integer notes to 5-dimensional dense vectors
- Multiple dilated `Conv1D` layers to capture local and long-range patterns
- `BatchNormalization` to stabilize training
- `Dropout` (5%) to prevent overfitting
- `LSTM` layer (256 units) to capture sequential dependencies
- Dense output layer (47 neurons, softmax) to predict next note probabilities

### 3. Training
- Optimizer: `Nadam` (Adam + Nesterov momentum), learning rate = 0.001
- Loss: `sparse_categorical_crossentropy`
- Metrics: accuracy
- Batch size: 32
- Ran on Google Colab GPU for faster training
- Validated on a separate validation set

### 4. Chorale Generation
- Use a small seed sequence of chords to prime the network
- Predict the next note at each timestep using the model's softmax output
- `sample_next_note` randomly selects notes according to predicted probabilities to introduce variation
- Concatenate predicted notes to the growing sequence
- Convert sequence back to MIDI notes
- Reshape for 4 voices and create a `music21` Stream for playback

## Final Output

- Generated chorales are stored as NumPy arrays
- Can be converted to `music21.stream.Stream` for MIDI playback
- Allows for different lengths and seed sequences to produce varied chorales

## Files in This Project

- `train_model.ipynb` - main notebook with preprocessing, training, and generation
- `chorales/` - folder with train, validation, and test CSV files
- `music_model.keras` - trained Keras model saved from Google Colab

## Timeline

9/15/25 - 9/21/25

## Future Improvements

- Experiment with transformer-based models for music generation  
- Add velocity and note duration modeling for more expressive music  
- Hyperparameter tuning for improved generation quality  
- Implement temperature sampling for more controlled randomness  
- Deploy a web interface to generate and play chorales interactively

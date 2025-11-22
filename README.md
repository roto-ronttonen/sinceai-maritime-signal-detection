### 1. Create Synthetic Training Data

Generate training dataset with variations in background noise, volume, and horn pairs:

```bash
uv run make_training_data.py

# Or in docker

docker compose up datagenerator
```

**Note**: Requires ffmpeg installed on system (`brew install ffmpeg` on macOS)

### 2. Train the YAMNet Classifier

Train the deep learning classifier using YAMNet embeddings:

```bash
uv run yamnet_classifier.py

## Or in docker

docker compsoe up trainclassifier
```

This will:

- Load the pre-trained YAMNet model from TensorFlow Hub
- Extract embeddings from all training samples
- Train a neural network classifier
- Save the model to `output/colreg_classifier.keras`

### 3. Make Predictions

Classify maritime horn signals in audio files:

```bash
# Single file prediction
uv run predict_signal.py path/to/audio.mp3

# Predict samples folder with docker
docker compose up classifier

```

### 4. Test the Implementation

Run the test suite:

```bash
uv run pytest test_yamnet_classifier.py
```

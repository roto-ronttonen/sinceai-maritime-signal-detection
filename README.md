### 1. Create Synthetic Training Data

Generate training dataset with variations in background noise, volume, and horn pairs:

```bash
uv run make_training_data.py

# Or in docker

docker compose up datagenerator
```

If you want to run the datageneration faster reduce the number of samples to generate (in code `num_samples_per_class = 250  # Lower num samples per class if you want to run it faster`).

**Note**: Requires ffmpeg installed on system (`brew install ffmpeg` on macOS)

### 2. Train the YAMNet Classifier

Train the deep learning classifier using YAMNet embeddings:

```bash
uv run yamnet_classifier.py

## Or in docker

docker compsoe up trainclassifier
```

If you want to train the classifier faster reduce the number of epochs. Currently it does 100 epochs which on a cpu take around 30 minutes.
Found in code in:

```
    history = classifier.train(
        X,
        y,
        test_size=0.2,  # 20% held out for final test
        val_size=0.15,  # 15% of training data for validation
        epochs=100,  # Will stop early if not improving. Lower epochs if you want to run it fast
        batch_size=32,  # Larger batch = better generalization
    )
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

# Or against all files in samples
uv run main.py

# Predict samples folder with docker
docker compose up classifier

```

### 4. Run everything in sequence

`./run_pipeline.sh`

### 5. Test the Implementation

Run the test suite:

```bash
uv run pytest test_yamnet_classifier.py
```

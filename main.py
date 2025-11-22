"""
Main script to classify maritime COLREG horn signals from audio files.
"""

import sys
import argparse
from pathlib import Path
from yamnet_classifier import YAMNetCOLREGClassifier


def main():
    """Process all audio files in the samples folder and predict COLREG signals."""
    parser = argparse.ArgumentParser(
        description="Classify maritime COLREG horn signals from audio files in samples folder"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="samples",
        help="Path to directory containing audio files (default: samples)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="output/colreg_classifier",
        help="Path to saved model (default: output/colreg_classifier)",
    )

    args = parser.parse_args()

    # Check if samples directory exists
    samples_path = Path(args.samples_dir)
    if not samples_path.exists():
        print(f"Error: Samples directory not found: {args.samples_dir}")
        sys.exit(1)

    # Check if model exists
    model_keras_path = Path(f"{args.model}.keras")
    if not model_keras_path.exists():
        print(f"Error: Model not found: {model_keras_path}")
        print(f"Please train the model first by running: uv run yamnet_classifier.py")
        sys.exit(1)

    # Find all audio files
    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    audio_files = [
        f
        for f in samples_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        print(f"Error: No audio files found in {args.samples_dir}")
        sys.exit(1)

    # Load the classifier
    print(f"Loading model from {args.model}...")
    classifier = YAMNetCOLREGClassifier()
    classifier.load_model(args.model)
    print(f"Found {len(audio_files)} audio file(s) to process\n")

    # Process each file
    for audio_file in sorted(audio_files):
        try:
            predicted_class, confidence, _ = classifier.predict(str(audio_file))
            print(f"{audio_file.name} - {predicted_class}")
        except Exception as e:
            print(f"{audio_file.name} - ERROR: {e}")


if __name__ == "__main__":
    main()

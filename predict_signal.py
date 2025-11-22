"""
Script to predict COLREG signal class from audio files using the trained YAMNet classifier.
"""

import sys
import argparse
from pathlib import Path
from yamnet_classifier import YAMNetCOLREGClassifier


def predict_audio_file(model_path, audio_file):
    """
    Predict the COLREG signal class for an audio file.

    Args:
        model_path: Path to the saved model (without extension)
        audio_file: Path to the audio file to classify
    """
    # Load the classifier
    print(f"Loading model from {model_path}...")
    classifier = YAMNetCOLREGClassifier()
    classifier.load_model(model_path)

    # Make prediction
    print(f"\nAnalyzing audio file: {audio_file}")
    predicted_class, confidence, probabilities = classifier.predict(audio_file)

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nPredicted Signal: {predicted_class.upper().replace('_', ' ')}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nAll Class Probabilities:")
    print("-" * 60)

    # Sort by probability for better readability
    sorted_results = sorted(
        zip(classifier.class_names, probabilities), key=lambda x: x[1], reverse=True
    )

    for class_name, prob in sorted_results:
        bar = "█" * int(prob * 50)
        print(f"{class_name:25s} {prob * 100:6.2f}% {bar}")

    print("=" * 60)

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict COLREG maritime horn signal from audio file"
    )
    parser.add_argument("audio_file", type=str, help="Path to audio file to classify")
    parser.add_argument(
        "--model",
        type=str,
        default="output/colreg_classifier",
        help="Path to saved model (default: output/colreg_classifier)",
    )

    args = parser.parse_args()

    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Check if model exists
    model_keras_path = Path(f"{args.model}.keras")
    if not model_keras_path.exists():
        print(f"Error: Model not found: {model_keras_path}")
        print(f"Please train the model first by running: uv run yamnet_classifier.py")
        sys.exit(1)

    # Make prediction
    try:
        predict_audio_file(args.model, str(audio_path))
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

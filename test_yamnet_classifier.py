"""
Pytest tests for YAMNet COLREG classifier.

Tests that each sample in the dataset is correctly predicted -
the top predicted class should match the actual label.
"""

import pytest
from pathlib import Path
from yamnet_classifier import YAMNetCOLREGClassifier


@pytest.fixture(scope="module")
def classifier():
    """Load the trained classifier once for all tests."""
    model_path = Path("output/colreg_classifier.keras")

    if not model_path.exists():
        pytest.skip("Model not found. Run 'uv run yamnet_classifier.py' first.")

    clf = YAMNetCOLREGClassifier()
    clf.load_model("output/colreg_classifier")
    return clf


def get_all_dataset_samples():
    """Get all audio files from dataset with their expected labels."""
    dataset_path = Path("dataset")

    if not dataset_path.exists():
        return []

    samples = []

    for class_dir in sorted(dataset_path.iterdir()):
        if class_dir.is_dir():
            expected_label = class_dir.name
            audio_files = list(class_dir.glob("*.mp3")) + list(class_dir.glob("*.wav"))

            for audio_file in sorted(audio_files):
                samples.append((str(audio_file), expected_label))

    return samples


# Generate parametrized test for each sample
@pytest.mark.parametrize("audio_file,expected_label", get_all_dataset_samples())
def test_predict_sample(classifier, audio_file, expected_label):
    """Test that the predicted class matches the expected label for each sample."""
    predicted_class, confidence, probabilities = classifier.predict(audio_file)

    assert predicted_class == expected_label, (
        f"Prediction failed for {Path(audio_file).name}\n"
        f"Expected: {expected_label}\n"
        f"Got: {predicted_class} (confidence: {confidence:.2%})"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

"""
YAMNet-based classifier for maritime COLREG horn signal sequences.

This module uses YAMNet (a pre-trained audio event classifier) as a feature extractor
and trains a classifier on top to recognize maritime horn signal patterns.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class YAMNetCOLREGClassifier:
    """Classifier for COLREG maritime horn signals using YAMNet embeddings."""

    def __init__(self, dataset_path="dataset", sample_rate=16000):
        """
        Initialize the classifier.

        Args:
            dataset_path: Path to the dataset folder containing class subdirectories
            sample_rate: Sample rate for audio processing (YAMNet uses 16kHz)
        """
        self.dataset_path = Path(dataset_path)
        self.sample_rate = sample_rate
        self.yamnet_model = None
        self.classifier_model = None
        self.label_encoder = {}
        self.class_names = []

    def load_yamnet(self):
        """Load the pre-trained YAMNet model from TensorFlow Hub."""
        print("Loading YAMNet model from TensorFlow Hub...")
        self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("YAMNet model loaded successfully!")

    def load_audio(self, file_path):
        """
        Load and preprocess audio file for YAMNet.

        Args:
            file_path: Path to audio file

        Returns:
            Audio waveform as numpy array at 16kHz sample rate
        """
        # Load audio file
        waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        # Normalize to [-1.0, 1.0]
        waveform = waveform / np.max(np.abs(waveform) + 1e-8)

        return waveform

    def extract_yamnet_embeddings(self, waveform, use_temporal=True):
        """
        Extract YAMNet embeddings from audio waveform.

        Args:
            waveform: Audio waveform (numpy array)
            use_temporal: If True, return full sequence. If False, return pooled features.

        Returns:
            If use_temporal=True: Sequence of embeddings (time_steps, 1024)
            If use_temporal=False: Combined mean and max pooled embeddings (2048-dim)
        """
        # YAMNet expects float32 tensor
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

        # Get embeddings from YAMNet
        # YAMNet returns: (scores, embeddings, spectrogram)
        _, embeddings, _ = self.yamnet_model(waveform_tensor)

        embeddings_np = embeddings.numpy()

        if use_temporal:
            # Return full sequence for temporal modeling
            return embeddings_np
        else:
            # Use both mean and max pooling for richer features
            mean_embedding = np.mean(embeddings_np, axis=0)
            max_embedding = np.max(embeddings_np, axis=0)

            # Concatenate for final feature vector
            combined_embedding = np.concatenate([mean_embedding, max_embedding])
            return combined_embedding

    def load_dataset(self, use_temporal=True):
        """
        Load the dataset and extract features.

        Args:
            use_temporal: If True, extract temporal sequences. If False, use pooled features.

        Returns:
            X: Feature matrix (sequences or pooled embeddings)
            y: Labels
            class_names: List of class names
        """
        if self.yamnet_model is None:
            self.load_yamnet()

        X = []
        y = []

        # Get all class directories
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        self.class_names = sorted([d.name for d in class_dirs])

        print(f"\nFound {len(self.class_names)} classes:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {i}: {class_name}")

        # Create label encoding
        self.label_encoder = {name: idx for idx, name in enumerate(self.class_names)}

        # Process each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            label = self.label_encoder[class_name]

            # Get all audio files in this class
            audio_files = list(class_dir.glob("*.mp3")) + list(class_dir.glob("*.wav"))

            print(f"\nProcessing {class_name}: {len(audio_files)} samples")

            for i, audio_file in enumerate(audio_files):
                try:
                    # Load audio
                    waveform = self.load_audio(audio_file)

                    # Extract embeddings
                    embedding = self.extract_yamnet_embeddings(
                        waveform, use_temporal=use_temporal
                    )

                    X.append(embedding)
                    y.append(label)

                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(audio_files)} files")

                except Exception as e:
                    print(f"  Error processing {audio_file}: {e}")
                    continue

        # Pad sequences if using temporal modeling
        if use_temporal and len(X) > 0:
            # Find max sequence length
            max_len = max(seq.shape[0] for seq in X)
            print(f"\nMax sequence length: {max_len} time steps")

            # Pad sequences to same length
            X_padded = []
            for seq in X:
                if seq.shape[0] < max_len:
                    # Pad with zeros
                    padding = np.zeros((max_len - seq.shape[0], seq.shape[1]))
                    padded_seq = np.vstack([seq, padding])
                else:
                    padded_seq = seq
                X_padded.append(padded_seq)

            X = np.array(X_padded)
            print(
                f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} time steps, {X.shape[2]} features"
            )
        else:
            X = np.array(X)
            print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

        y = np.array(y)
        print(f"Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label_idx, count in zip(unique, counts):
            print(f"  {self.class_names[label_idx]}: {count}")

        return X, y, self.class_names

    def build_classifier(self, input_shape, num_classes, use_temporal=True):
        """
        Build a neural network classifier on top of YAMNet embeddings.

        Args:
            input_shape: Shape of input (time_steps, features) or (features,)
            num_classes: Number of output classes
            use_temporal: If True, build LSTM model. If False, build dense model.

        Returns:
            Compiled Keras model
        """
        if use_temporal:
            # LSTM-based temporal model
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=input_shape),
                    # Masking layer to handle padded sequences
                    tf.keras.layers.Masking(mask_value=0.0),
                    # First LSTM layer - bidirectional for better context
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            128,
                            return_sequences=True,
                            dropout=0.3,
                            recurrent_dropout=0.2,
                        )
                    ),
                    tf.keras.layers.BatchNormalization(),
                    # Second LSTM layer
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            64,
                            return_sequences=False,
                            dropout=0.3,
                            recurrent_dropout=0.2,
                        )
                    ),
                    tf.keras.layers.BatchNormalization(),
                    # Dense layers for final classification
                    tf.keras.layers.Dense(
                        128,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    ),
                    tf.keras.layers.Dropout(0.4),
                    tf.keras.layers.Dense(
                        64,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    ),
                    tf.keras.layers.Dropout(0.3),
                    # Output layer
                    tf.keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )
        else:
            # Dense model for pooled features
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(input_shape,)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(
                        512,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(
                        256,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.4),
                    tf.keras.layers.Dense(
                        128,
                        activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    ),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(self, X, y, test_size=0.2, epochs=150, batch_size=16):
        """
        Train the classifier.

        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of data for testing
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Build model
        num_classes = len(self.class_names)
        use_temporal = len(X.shape) == 3  # 3D shape means temporal sequences

        if use_temporal:
            input_shape = (X.shape[1], X.shape[2])  # (time_steps, features)
        else:
            input_shape = X.shape[1]  # just features

        self.classifier_model = self.build_classifier(
            input_shape, num_classes, use_temporal=use_temporal
        )

        print("\nModel architecture:")
        self.classifier_model.summary()

        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
        )

        # Train
        print("\nTraining classifier...")
        history = self.classifier_model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        # Evaluate
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.classifier_model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Predictions and metrics
        y_pred = np.argmax(self.classifier_model.predict(X_test), axis=1)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("Confusion matrix saved to output/confusion_matrix.png")
        plt.close()

    def plot_training_history(self, history):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        ax1.plot(history.history["accuracy"], label="Train Accuracy")
        ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(history.history["loss"], label="Train Loss")
        ax2.plot(history.history["val_loss"], label="Val Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("output/training_history.png", dpi=300, bbox_inches="tight")
        print("Training history saved to output/training_history.png")
        plt.close()

    def save_model(self, model_path="output/colreg_classifier"):
        """
        Save the trained classifier and metadata.

        Args:
            model_path: Base path for saving model files
        """
        os.makedirs("output", exist_ok=True)

        # Save Keras model
        self.classifier_model.save(f"{model_path}.keras")
        print(f"Model saved to {model_path}.keras")

        # Save metadata
        metadata = {
            "class_names": self.class_names,
            "label_encoder": self.label_encoder,
            "sample_rate": self.sample_rate,
        }
        joblib.dump(metadata, f"{model_path}_metadata.pkl")
        print(f"Metadata saved to {model_path}_metadata.pkl")

    def load_model(self, model_path="output/colreg_classifier"):
        """
        Load a trained classifier.

        Args:
            model_path: Base path for loading model files
        """
        # Load Keras model
        self.classifier_model = tf.keras.models.load_model(f"{model_path}.keras")
        print(f"Model loaded from {model_path}.keras")

        # Load metadata
        metadata = joblib.load(f"{model_path}_metadata.pkl")
        self.class_names = metadata["class_names"]
        self.label_encoder = metadata["label_encoder"]
        self.sample_rate = metadata["sample_rate"]
        print(f"Metadata loaded from {model_path}_metadata.pkl")

        # Load YAMNet for inference
        if self.yamnet_model is None:
            self.load_yamnet()

    def predict(self, audio_file_path):
        """
        Predict the class of an audio file.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        if self.yamnet_model is None:
            self.load_yamnet()

        if self.classifier_model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        # Load and process audio
        waveform = self.load_audio(audio_file_path)

        # Determine if model uses temporal sequences
        use_temporal = len(self.classifier_model.input_shape) == 3

        # Extract embeddings
        embedding = self.extract_yamnet_embeddings(waveform, use_temporal=use_temporal)

        # Pad if necessary for temporal model
        if use_temporal:
            expected_time_steps = self.classifier_model.input_shape[1]
            if embedding.shape[0] < expected_time_steps:
                padding = np.zeros(
                    (expected_time_steps - embedding.shape[0], embedding.shape[1])
                )
                embedding = np.vstack([embedding, padding])
            elif embedding.shape[0] > expected_time_steps:
                embedding = embedding[:expected_time_steps]

        # Predict
        probabilities = self.classifier_model.predict(np.array([embedding]), verbose=0)[
            0
        ]
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx]

        return predicted_class, confidence, probabilities


def main():
    """Main function to train the YAMNet COLREG classifier."""
    print("=" * 80)
    print("YAMNet-based COLREG Maritime Horn Signal Classifier")
    print("=" * 80)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Initialize classifier
    classifier = YAMNetCOLREGClassifier(dataset_path="dataset")

    # Load dataset and extract features (with temporal modeling)
    X, y, class_names = classifier.load_dataset(use_temporal=True)

    # Train classifier
    ## TODO set epochs to 100
    history = classifier.train(X, y, test_size=0.2, epochs=5, batch_size=16)

    # Save model
    classifier.save_model("output/colreg_classifier")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Model saved to: output/colreg_classifier.keras")
    print(f"Metadata saved to: output/colreg_classifier_metadata.pkl")
    print(f"Confusion matrix: output/confusion_matrix.png")
    print(f"Training history: output/training_history.png")

    # Example prediction on a sample file
    print("\n" + "=" * 80)
    print("Testing prediction on a sample file...")
    print("=" * 80)

    # Find a sample file
    sample_dir = Path("dataset/turn_port")
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.mp3"))
        if sample_files:
            sample_file = sample_files[0]
            predicted_class, confidence, probabilities = classifier.predict(sample_file)

            print(f"\nSample file: {sample_file}")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            print("\nAll class probabilities:")
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()

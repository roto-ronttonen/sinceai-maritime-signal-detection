"""
Diagnostic tool to analyze what's being detected in audio files.
Helps understand why predictions are failing.
"""

import numpy as np
import librosa
from scipy import signal as scipy_signal
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_audio(audio_path: str):
    """Analyze an audio file and show what's being detected."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(audio_path).name}")
    print(f"{'='*80}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-8)

    # Bandpass filter
    nyquist = sr / 2
    sos = scipy_signal.butter(
        4, [200 / nyquist, 2000 / nyquist], btype="band", output="sos"
    )
    y_filtered = scipy_signal.sosfilt(sos, y)

    # Energy envelope
    hop_length = int(sr * 0.01)
    rms = librosa.feature.rms(
        y=y_filtered, frame_length=hop_length * 4, hop_length=hop_length
    )[0]
    rms = rms / (np.max(rms) + 1e-8)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Try different thresholds
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        is_active = rms > threshold

        # Detect blasts
        blast_durations = []
        in_blast = False
        start_time = 0

        for i, active in enumerate(is_active):
            if active and not in_blast:
                start_time = times[i]
                in_blast = True
            elif not active and in_blast:
                duration = times[i] - start_time
                if duration >= 0.5:
                    blast_durations.append(duration)
                in_blast = False

        num_short = sum(1 for d in blast_durations if d < 3.0)
        num_long = sum(1 for d in blast_durations if d >= 3.0)

        print(f"\nThreshold {threshold}:")
        print(f"  Total blasts: {len(blast_durations)}")
        print(f"  Short blasts: {num_short}")
        print(f"  Long blasts: {num_long}")
        if blast_durations:
            print(f"  Durations: {[f'{d:.2f}s' for d in blast_durations]}")

    # Statistics
    print(f"\nAudio stats:")
    print(f"  Duration: {len(y)/sr:.2f}s")
    print(f"  RMS mean: {np.mean(rms):.3f}")
    print(f"  RMS max: {np.max(rms):.3f}")
    print(f"  RMS std: {np.std(rms):.3f}")

    # Plot for visualization
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(times, rms)
    plt.axhline(y=0.3, color="r", linestyle="--", label="Threshold 0.3")
    plt.axhline(y=0.2, color="orange", linestyle="--", label="Threshold 0.2")
    plt.title(f"Energy Envelope: {Path(audio_path).name}")
    plt.ylabel("Normalized RMS Energy")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(y)) / sr, y_filtered)
    plt.title("Filtered Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    output_file = f"debug_{Path(audio_path).stem}.png"
    plt.savefig(output_file)
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def main():
    """Analyze sample files to debug detection."""
    import sys

    if len(sys.argv) > 1:
        # Analyze specific file
        audio_path = sys.argv[1]
        analyze_audio(audio_path)
    else:
        # Analyze first few samples
        samples_dir = Path("samples")
        audio_files = sorted(
            list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
        )

        for audio_file in audio_files[:5]:
            analyze_audio(str(audio_file))
            print()


if __name__ == "__main__":
    main()

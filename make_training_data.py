import os
import shutil
from pydub import AudioSegment
import random

signal_folder = "signals"
background_folder = "backgrounds"
output_folder = "dataset"

shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)

pair_folders = [
    os.path.join(signal_folder, d)
    for d in os.listdir(signal_folder)
    if os.path.isdir(os.path.join(signal_folder, d))
]
horn_pairs = []
for pf in pair_folders:
    short_file_mp3 = os.path.join(pf, "short.mp3")
    long_file_mp3 = os.path.join(pf, "long.mp3")
    short_file_wav = os.path.join(pf, "short.wav")
    long_file_wav = os.path.join(pf, "long.wav")

    short_file = short_file_mp3 if os.path.exists(short_file_mp3) else short_file_wav
    long_file = long_file_mp3 if os.path.exists(long_file_mp3) else long_file_wav

    if os.path.exists(short_file) and os.path.exists(long_file):
        horn_pairs.append((short_file, long_file))

background_files = [
    os.path.join(background_folder, f)
    for f in os.listdir(background_folder)
    if f.endswith(".mp3") or f.endswith(".wav")
]
backgrounds = [AudioSegment.from_file(f) for f in background_files]


short_duration_ms = 1000
long_duration_ms = 5000
silence_between_ms = 1000
tiny_break_ms = 1500

# Increased variation parameters for better generalization
volume_variation_db = 6  # Increased from 2 - more realistic volume differences
background_amplification_db = 2
background_variation_db = 4  # Additional variation in background levels
num_samples_per_class = 250  # Lower num samples per class if you want to run it faster
target_duration_range_ms = (25000, 30000)
horn_start_delay_range_ms = (1000, 10000)

# New parameters for more variation
horn_duration_variation_ms = 250  # Vary horn blast lengths
silence_variation_ms = 200  # Vary silence between blasts
tiny_break_variation_ms = 200  # Vary the tiny break duration


def random_volume(audio, variation_db=10):
    change = random.uniform(-variation_db, variation_db)
    return audio + change


def add_background(signal, backgrounds, amplification_db=5):
    """Add background with variation in amplification level."""
    # Vary background amplification more for diversity
    bg_variation = random.uniform(-background_variation_db, background_variation_db)
    bg = random.choice(backgrounds) + amplification_db + bg_variation
    if len(bg) < len(signal):
        bg = bg * (len(signal) // len(bg) + 1)
    bg = bg[: len(signal)]
    return signal.overlay(bg)


def pad_with_background(audio, target_duration_ms, backgrounds, amplification_db=5):
    """Extend or trim audio to target_duration_ms, maintaining background throughout"""
    current_duration = len(audio)

    if current_duration < target_duration_ms:
        # Create continuous background for the entire target duration
        padding_needed = target_duration_ms - current_duration
        bg_variation = random.uniform(-background_variation_db, background_variation_db)
        bg = random.choice(backgrounds) + amplification_db + bg_variation
        # Loop background if needed
        if len(bg) < padding_needed:
            bg = bg * (padding_needed // len(bg) + 1)
        bg_padding = bg[:padding_needed]
        # Overlay padding background to continue seamlessly
        audio = audio + bg_padding
    elif current_duration > target_duration_ms:
        # Trim from the end
        audio = audio[:target_duration_ms]

    return audio


def generate_sequence(pattern, short_horn_file, long_horn_file, start_delay_ms):
    # Add duration variation to horn blasts for more realism
    short_dur = short_duration_ms + random.randint(
        -horn_duration_variation_ms, horn_duration_variation_ms
    )
    long_dur = long_duration_ms + random.randint(
        -horn_duration_variation_ms, horn_duration_variation_ms
    )

    # Ensure minimum durations
    short_dur = max(600, short_dur)
    long_dur = max(3500, long_dur)

    short_horn = AudioSegment.from_file(short_horn_file)[:short_dur]
    long_horn = AudioSegment.from_file(long_horn_file)[:long_dur]

    # Build horn sequence
    horn_sequence = AudioSegment.empty()
    for i, blast in enumerate(pattern):
        if blast == "short":
            clip = random_volume(short_horn, volume_variation_db)
            horn_sequence += clip
        elif blast == "long":
            clip = random_volume(long_horn, volume_variation_db)
            horn_sequence += clip
        elif blast == "tiny_break":
            # Vary tiny break duration
            varied_break = tiny_break_ms + random.randint(
                -tiny_break_variation_ms, tiny_break_variation_ms
            )
            varied_break = max(1000, varied_break)  # Minimum 1 second
            horn_sequence += AudioSegment.silent(duration=varied_break)

        if i < len(pattern) - 1 and blast != "tiny_break":
            # Vary silence between blasts
            varied_silence = silence_between_ms + random.randint(
                -silence_variation_ms, silence_variation_ms
            )
            varied_silence = max(500, varied_silence)  # Minimum 0.5 seconds
            horn_sequence += AudioSegment.silent(duration=varied_silence)

    # Create background that covers intro + horn sequence duration
    total_duration_needed = start_delay_ms + len(horn_sequence)
    bg_variation = random.uniform(-background_variation_db, background_variation_db)
    bg = random.choice(backgrounds) + background_amplification_db + bg_variation
    if len(bg) < total_duration_needed:
        bg = bg * (total_duration_needed // len(bg) + 1)
    bg = bg[:total_duration_needed]

    # Overlay the horn sequence at the start_delay position
    # Use mix=True to ensure background continues through and after the horn
    sequence = bg.overlay(horn_sequence, position=start_delay_ms)

    return sequence


colreg_sequences = {
    "altering_starboard": ["short"],
    "altering_port": ["short", "short"],
    "astern_propagation": ["short", "short", "short"],
    "turn_starboard": ["short", "short", "short", "short", "tiny_break", "short"],
    "turn_port": ["short", "short", "short", "short", "tiny_break", "short", "short"],
    "do_not_understand": ["short", "short", "short", "short", "short"],
    "about_underway": ["long"],
    "unable_to_manoeuvre": ["long", "short", "short"],
    "overtake_starboard": ["long", "long", "short"],
    "overtake_port": ["long", "long", "short", "short"],
    "agree_to_be_overtaken": ["long", "short", "long", "short"],
}


for label, pattern in colreg_sequences.items():
    label_folder = os.path.join(output_folder, label)
    os.makedirs(label_folder, exist_ok=True)
    for i in range(num_samples_per_class):
        short_horn_file, long_horn_file = random.choice(horn_pairs)

        # Random delay before horn starts (1-10 seconds)
        start_delay = random.randint(*horn_start_delay_range_ms)
        sequence = generate_sequence(
            pattern, short_horn_file, long_horn_file, start_delay
        )

        # Pad to 25-30 seconds with background noise
        target_duration = random.randint(*target_duration_range_ms)
        sequence = pad_with_background(
            sequence, target_duration, backgrounds, background_amplification_db
        )

        filename = os.path.join(label_folder, f"{label}_{i}.mp3")
        sequence.export(filename, format="mp3", bitrate="192k")
        print(f"Saved {filename}")


no_signal_folder = os.path.join(output_folder, "no_signal")
os.makedirs(no_signal_folder, exist_ok=True)
for i in range(num_samples_per_class):
    # Generate background noise with varying content and levels
    target_duration = random.randint(*target_duration_range_ms)

    bg = random.choice(backgrounds)
    # Loop background if needed to reach target duration
    if len(bg) < target_duration:
        bg = bg * (target_duration // len(bg) + 1)
    bg = bg[:target_duration]

    # Apply varied amplification to background
    bg_variation = random.uniform(-background_variation_db, background_variation_db)
    bg = bg + background_amplification_db + bg_variation
    bg = random_volume(bg, volume_variation_db)

    filename = os.path.join(no_signal_folder, f"no_signal_{i}.mp3")
    bg.export(filename, format="mp3", bitrate="192k")
    print(f"Saved {filename}")

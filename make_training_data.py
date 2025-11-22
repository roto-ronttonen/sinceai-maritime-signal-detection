import os
import shutil
from pydub import AudioSegment
import random

# --- Config ---
signal_folder = "signals"
background_folder = "backgrounds"
output_folder = "dataset"

shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)

# Load horn pairs
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
# Load backgrounds
background_files = [
    os.path.join(background_folder, f)
    for f in os.listdir(background_folder)
    if f.endswith(".mp3") or f.endswith(".wav")
]
backgrounds = [AudioSegment.from_file(f) for f in background_files]

# Parameters
short_duration_ms = 1000
long_duration_ms = 5000
silence_between_ms = 1000
tiny_break_ms = 1500
volume_variation_db = 10
background_amplification_db = 5  # How much to amplify background sounds
num_samples_per_class = 101
pre_horn_silence_range = (2000, 4000)  # 0–1s random silence at start
post_horn_silence_range = (2000, 4000)  # 0–1s random silence at end


# --- Helper functions ---
def random_volume(audio, variation_db=10):
    change = random.uniform(-variation_db, variation_db)
    return audio + change


def add_background(signal, backgrounds, amplification_db=5):
    bg = random.choice(backgrounds) + amplification_db
    if len(bg) < len(signal):
        bg = bg * (len(signal) // len(bg) + 1)
    bg = bg[: len(signal)]
    return signal.overlay(bg)


def generate_sequence(pattern, short_horn_file, long_horn_file):
    short_horn = AudioSegment.from_file(short_horn_file)[:short_duration_ms]
    long_horn = AudioSegment.from_file(long_horn_file)[:long_duration_ms]

    # Random pre-horn silence
    sequence = AudioSegment.silent(duration=random.randint(*pre_horn_silence_range))

    # Add horn blasts
    for i, blast in enumerate(pattern):
        if blast == "short":
            clip = random_volume(short_horn, volume_variation_db)
            sequence += clip
        elif blast == "long":
            clip = random_volume(long_horn, volume_variation_db)
            sequence += clip
        elif blast == "tiny_break":
            sequence += AudioSegment.silent(duration=tiny_break_ms)

        # Add normal interval (1s) between blasts
        if i < len(pattern) - 1 and blast != "tiny_break":
            sequence += AudioSegment.silent(duration=silence_between_ms)

    # Random post-horn silence
    sequence += AudioSegment.silent(duration=random.randint(*post_horn_silence_range))

    # Add background noise
    sequence = add_background(sequence, backgrounds, background_amplification_db)
    return sequence


# --- COLREG sequences ---
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

# --- Generate dataset ---
for label, pattern in colreg_sequences.items():
    label_folder = os.path.join(output_folder, label)
    os.makedirs(label_folder, exist_ok=True)
    for i in range(num_samples_per_class):
        short_horn_file, long_horn_file = random.choice(horn_pairs)
        sequence = generate_sequence(pattern, short_horn_file, long_horn_file)
        filename = os.path.join(label_folder, f"{label}_{i}.mp3")
        sequence.export(filename, format="mp3", bitrate="192k")
        print(f"Saved {filename}")

# --- Generate 'no_signal' negative examples ---
no_signal_folder = os.path.join(output_folder, "no_signal")
os.makedirs(no_signal_folder, exist_ok=True)
for i in range(num_samples_per_class):
    # Limit duration to ~max length of a signal sequence
    max_duration_ms = 7000  # adjust according to your sequences
    duration_ms = random.randint(int(max_duration_ms * 0.8), max_duration_ms)

    bg = random.choice(backgrounds)[:duration_ms]

    # Amplify background to match signal sequences
    bg = bg + background_amplification_db

    # Add random pre- and post-silence to match signal sequences
    pre_silence = AudioSegment.silent(duration=random.randint(*pre_horn_silence_range))
    post_silence = AudioSegment.silent(
        duration=random.randint(*post_horn_silence_range)
    )
    bg = pre_silence + bg + post_silence

    bg = random_volume(bg, volume_variation_db)
    filename = os.path.join(no_signal_folder, f"no_signal_{i}.mp3")
    bg.export(filename, format="mp3", bitrate="192k")
    print(f"Saved {filename}")

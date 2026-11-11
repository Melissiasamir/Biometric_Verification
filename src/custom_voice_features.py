
from pathlib import Path
import csv

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "custom_dataset" / "voice"
OUTPUT_DIR = PROJECT_ROOT / "features" / "custom_voice"

FEATURES_FILE = OUTPUT_DIR / "custom_voice_features.npy"
METADATA_FILE = OUTPUT_DIR / "custom_voice_metadata.csv"

SAMPLE_RATE = 32000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512


def extract_voice_features(audio_path):
    audio, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = []

    for feature_matrix in [mfcc, delta, delta2]:
        features.extend(np.mean(feature_matrix, axis=1))
        features.extend(np.std(feature_matrix, axis=1))

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.extend([
        np.mean(zcr),
        np.std(zcr),
        np.mean(rms),
        np.std(rms)
    ])

    return np.asarray(features, dtype=np.float32)


def main():
    print("\n========== CUSTOM VOICE FEATURE EXTRACTION ==========\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []
    metadata = []

    for person_dir in sorted(DATA_DIR.iterdir()):

        if not person_dir.is_dir():
            continue

        audio_files = sorted(person_dir.glob("*.wav"))

        if not audio_files:
            continue

        print(
            f"\nProcessing {person_dir.name}: "
            f"{len(audio_files)} recordings"
        )

        for audio_path in audio_files:

            features = extract_voice_features(audio_path)

            all_features.append(features)

            metadata.append({
                "Person_ID": person_dir.name,
                "Sample_ID": audio_path.stem,
                "Audio_Path": str(
                    audio_path.relative_to(PROJECT_ROOT)
                )
            })

            print(f"Extracted: {audio_path.name}")

    if not all_features:
        raise RuntimeError("No voice recordings found.")

    feature_matrix = np.vstack(all_features)

    np.save(FEATURES_FILE, feature_matrix)

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Person_ID",
                "Sample_ID",
                "Audio_Path"
            ]
        )

        writer.writeheader()
        writer.writerows(metadata)

    print("\n========== EXTRACTION SUMMARY ==========")
    print(f"Total recordings: {len(metadata)}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"All features finite: {np.isfinite(feature_matrix).all()}")
    print(f"Features saved: {FEATURES_FILE}")
    print(f"Metadata saved: {METADATA_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()

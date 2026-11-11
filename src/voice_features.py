
import os
import numpy as np
import pandas as pd
import librosa

from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

SPLITS_PATH = BASE_DIR / "data" / "vidtimit" / "splits.csv"

OUTPUT_DIR = BASE_DIR / "features" / "voice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = OUTPUT_DIR / "voice_features.npy"
METADATA_PATH = OUTPUT_DIR / "voice_metadata.csv"


def extract_voice_features(audio_path):
    """
    Extract 82 voice features:
    - MFCC mean and standard deviation: 26
    - MFCC Delta mean and standard deviation: 26
    - MFCC Delta-Delta mean and standard deviation: 26
    - ZCR mean and standard deviation: 2
    - RMS mean and standard deviation: 2
    Total = 82 features
    """

    signal, sr = librosa.load(
        audio_path,
        sr=32000,
        mono=True
    )

    if len(signal) == 0:
        raise ValueError("Empty audio signal")

    # Static MFCC features
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    # First-order MFCC derivatives
    mfcc_delta = librosa.feature.delta(mfcc, order=1)

    # Second-order MFCC derivatives
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)

    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=2048,
        hop_length=512
    )

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Root Mean Square Energy
    rms = librosa.feature.rms(
        y=signal,
        frame_length=2048,
        hop_length=512
    )

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Combine all features
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std,
        delta2_mean,
        delta2_std,
        [zcr_mean, zcr_std],
        [rms_mean, rms_std]
    ])

    if features.shape[0] != 82:
        raise ValueError(
            f"Expected 82 features, got {features.shape[0]}"
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("Extracted features contain NaN or infinity")

    return features


def main():

    print("Loading split information...")

    splits_df = pd.read_csv(SPLITS_PATH)

    features_list = []
    metadata_list = []

    total = len(splits_df)
    successful = 0
    failed = 0

    print(f"Total recordings: {total}")
    print("Starting voice feature extraction...\n")

    for index, row in splits_df.iterrows():

        audio_path = row["Audio_Path"]

        try:
            features = extract_voice_features(audio_path)

            features_list.append(features)

            metadata_list.append({
                "Person_ID": row["Person_ID"],
                "Sequence_ID": row["Sequence_ID"],
                "Split": row["Split"],
                "Audio_File": os.path.basename(audio_path)
            })

            successful += 1

        except Exception as error:
            failed += 1

            print(
                f"Failed: {row['Person_ID']} / "
                f"{row['Sequence_ID']}: {error}"
            )

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{total}")

    if successful == 0:
        raise RuntimeError("No voice features were extracted successfully")

    # Convert to arrays and DataFrame
    feature_matrix = np.array(features_list, dtype=np.float64)
    metadata_df = pd.DataFrame(metadata_list)

    # Save outputs
    np.save(FEATURES_PATH, feature_matrix)
    metadata_df.to_csv(METADATA_PATH, index=False)

    print("\n========== VOICE EXTRACTION RESULTS ==========")
    print(f"Total recordings: {total}")
    print(f"Successfully extracted: {successful}")
    print(f"Failed recordings: {failed}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"All features finite: {np.all(np.isfinite(feature_matrix))}")

    print(f"\nFeatures saved to: {FEATURES_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()

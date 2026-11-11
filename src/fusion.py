
from pathlib import Path
import csv

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACE_FEATURES_FILE = (
    PROJECT_ROOT / "features" / "face" / "face_features.npy"
)
VOICE_FEATURES_FILE = (
    PROJECT_ROOT / "features" / "voice" / "voice_features.npy"
)

FACE_METADATA_FILE = (
    PROJECT_ROOT / "features" / "face" / "face_metadata.csv"
)
VOICE_METADATA_FILE = (
    PROJECT_ROOT / "features" / "voice" / "voice_metadata.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "features" / "fusion"

# Best voice configuration from verification experiments:
# 26 static MFCC features + 2 ZCR features + 2 RMS features.
VOICE_FEATURE_COLUMNS = (
    list(range(0, 26)) + [78, 79, 80, 81]
)


def load_metadata(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_inputs(
    face_features,
    voice_features,
    face_metadata,
    voice_metadata
):
    if len(face_metadata) != len(voice_metadata):
        raise ValueError("Metadata row counts do not match.")

    if face_features.shape[0] != len(face_metadata):
        raise ValueError(
            "Face feature and metadata counts do not match."
        )

    if voice_features.shape[0] != len(voice_metadata):
        raise ValueError(
            "Voice feature and metadata counts do not match."
        )

    if face_features.shape[0] != voice_features.shape[0]:
        raise ValueError(
            "Face and Voice feature counts do not match."
        )

    face_keys = [
        (row["Person_ID"], row["Sequence_ID"], row["Split"])
        for row in face_metadata
    ]

    voice_keys = [
        (row["Person_ID"], row["Sequence_ID"], row["Split"])
        for row in voice_metadata
    ]

    if face_keys != voice_keys:
        raise ValueError(
            "Face and Voice recording alignment failed."
        )

    if voice_features.ndim != 2 or face_features.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional.")

    if voice_features.shape[1] <= max(VOICE_FEATURE_COLUMNS):
        raise ValueError(
            "Voice feature array does not contain the required columns."
        )

    if not np.isfinite(face_features).all():
        raise ValueError("Face features contain invalid values.")

    if not np.isfinite(voice_features).all():
        raise ValueError("Voice features contain invalid values.")


def main():
    print("Loading Face and Voice features...")

    face_features = np.load(FACE_FEATURES_FILE)
    voice_features = np.load(VOICE_FEATURES_FILE)

    face_metadata = load_metadata(FACE_METADATA_FILE)
    voice_metadata = load_metadata(VOICE_METADATA_FILE)

    print("Validating input data...")

    validate_inputs(
        face_features,
        voice_features,
        face_metadata,
        voice_metadata
    )

    print("Input validation passed.")

    print("Selecting best Voice feature configuration...")

    selected_voice_features = voice_features[
        :, VOICE_FEATURE_COLUMNS
    ]

    print("Performing Early Fusion...")

    fusion_features = np.concatenate(
        [face_features, selected_voice_features],
        axis=1
    ).astype(np.float32)

    if not np.isfinite(fusion_features).all():
        raise ValueError(
            "Fusion features contain invalid values."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = OUTPUT_DIR / "fusion_features.npy"
    metadata_path = OUTPUT_DIR / "fusion_metadata.csv"

    np.save(features_path, fusion_features)

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Person_ID",
            "Sequence_ID",
            "Split"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in face_metadata:
            writer.writerow({
                "Person_ID": row["Person_ID"],
                "Sequence_ID": row["Sequence_ID"],
                "Split": row["Split"]
            })

    print("\n========== EARLY FUSION REPORT ==========")
    print("Face feature shape:", face_features.shape)
    print("Original Voice feature shape:", voice_features.shape)
    print("Selected Voice feature shape:", selected_voice_features.shape)
    print("Fusion feature shape:", fusion_features.shape)
    print("Total recordings:", len(face_metadata))
    print("Features per recording:", fusion_features.shape[1])
    print("All features finite:", np.isfinite(fusion_features).all())
    print("Recording alignment: PASSED")
    print("Split alignment: PASSED")
    print("Fusion features saved to:", features_path)
    print("Fusion metadata saved to:", metadata_path)
    print("=========================================")


if __name__ == "__main__":
    main()

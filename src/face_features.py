
from pathlib import Path
import csv

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_FILE = PROJECT_ROOT / "data" / "vidtimit" / "splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "features" / "face"
FEATURES_FILE = OUTPUT_DIR / "face_features.npy"
METADATA_FILE = OUTPUT_DIR / "face_metadata.csv"

# Feature extraction settings
IMAGE_SIZE = (128, 128)
SAMPLE_FRAMES = 10

HOG_ORIENTATIONS = 9
LBP_POINTS = 8
LBP_RADIUS = 1

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def get_frame_files(face_dir):
    files = [
        p for p in Path(face_dir).iterdir()
        if p.is_file()
    ]

    return sorted(files, key=lambda p: int(p.name.split(".")[0]))


def extract_frame_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    x, y, width, height = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    face = gray[y:y + height, x:x + width]
    face = cv2.resize(face, IMAGE_SIZE)

    # HOG
    hog_features = hog(
        face,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    # LBP
    lbp = local_binary_pattern(
        face,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )

    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    return np.concatenate([
        hog_features,
        lbp_hist
    ]).astype(np.float32)


def extract_sequence_features(record):
    frame_files = get_frame_files(record["Face_Path"])

    if len(frame_files) < SAMPLE_FRAMES:
        return None, 0, SAMPLE_FRAMES

    indices = np.linspace(
        0,
        len(frame_files) - 1,
        SAMPLE_FRAMES,
        dtype=int
    )

    selected_frames = [frame_files[i] for i in indices]

    frame_features = []

    for frame_path in selected_frames:
        features = extract_frame_features(frame_path)

        if features is not None:
            frame_features.append(features)

    successful = len(frame_features)
    failed = SAMPLE_FRAMES - successful

    if successful == 0:
        return None, successful, failed

    frame_features = np.vstack(frame_features)

    mean_features = np.mean(frame_features, axis=0)
    std_features = np.std(frame_features, axis=0)

    sequence_features = np.concatenate([
        mean_features,
        std_features
    ]).astype(np.float32)

    return sequence_features, successful, failed


def create_face_features():
    if FACE_CASCADE.empty():
        raise RuntimeError("Face detection cascade could not be loaded.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLITS_FILE, "r", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    all_features = []
    metadata = []

    failed_sequences = []
    partial_sequences = []

    print("Starting face feature extraction...")
    print("Total sequences:", len(records))

    for index, record in enumerate(records, start=1):

        features, successful, failed = extract_sequence_features(record)

        if features is None:
            failed_sequences.append(
                (record["Person_ID"], record["Sequence_ID"])
            )
            print(
                f"[{index}/{len(records)}] FAILED: "
                f"{record['Person_ID']} - {record['Sequence_ID']}"
            )
            continue

        all_features.append(features)

        metadata.append({
            "Person_ID": record["Person_ID"],
            "Sequence_ID": record["Sequence_ID"],
            "Split": record["Split"],
            "Successful_Frames": successful,
            "Failed_Frames": failed
        })

        if failed > 0:
            partial_sequences.append(
                (record["Person_ID"], record["Sequence_ID"], failed)
            )

        if index % 25 == 0 or index == len(records):
            print(f"Processed: {index}/{len(records)}")

    if not all_features:
        raise RuntimeError("No face features were extracted.")

    feature_matrix = np.vstack(all_features)

    np.save(FEATURES_FILE, feature_matrix)

    with open(
        METADATA_FILE, "w", newline="", encoding="utf-8"
    ) as file:

        fieldnames = [
            "Person_ID",
            "Sequence_ID",
            "Split",
            "Successful_Frames",
            "Failed_Frames"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print("\n========== FACE EXTRACTION SUMMARY ==========")
    print("Sequences in manifest:", len(records))
    print("Successfully extracted:", len(all_features))
    print("Completely failed:", len(failed_sequences))
    print("Sequences with partial detection:", len(partial_sequences))
    print("Feature matrix shape:", feature_matrix.shape)
    print("All features finite:", np.isfinite(feature_matrix).all())
    print("Features saved to:", FEATURES_FILE)
    print("Metadata saved to:", METADATA_FILE)

    if failed_sequences:
        print("\nFailed sequences:", failed_sequences)

    if partial_sequences:
        print("\nPartial detection examples:", partial_sequences[:10])


if __name__ == "__main__":
    create_face_features()

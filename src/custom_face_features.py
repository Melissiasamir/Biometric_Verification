from pathlib import Path
import csv

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "custom_dataset" / "face"
OUTPUT_DIR = PROJECT_ROOT / "features" / "custom_face"

FEATURES_FILE = OUTPUT_DIR / "custom_face_features.npy"
METADATA_FILE = OUTPUT_DIR / "custom_face_metadata.csv"

IMAGE_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
LBP_POINTS = 8
LBP_RADIUS = 1


def extract_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)

    hog_features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    lbp = local_binary_pattern(
        gray,
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
    lbp_hist /= lbp_hist.sum() + 1e-7

    return np.concatenate([
        hog_features,
        lbp_hist
    ]).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []
    metadata = []

    print("========== CUSTOM FACE FEATURE EXTRACTION ==========")

    for person_dir in sorted(DATA_DIR.iterdir()):

        if not person_dir.is_dir():
            continue

        image_files = sorted(person_dir.glob("*.jpg"))

        if not image_files:
            print(f"No images found for {person_dir.name}")
            continue

        print(f"\nProcessing {person_dir.name}: {len(image_files)} images")

        for image_path in image_files:

            features = extract_features(image_path)

            all_features.append(features)

            metadata.append({
                "Person_ID": person_dir.name,
                "Sample_ID": image_path.stem,
                "Image_Path": str(image_path.relative_to(PROJECT_ROOT))
            })

            print(f"Extracted: {image_path.name}")

    if not all_features:
        raise RuntimeError("No custom face features were extracted.")

    feature_matrix = np.vstack(all_features)

    np.save(FEATURES_FILE, feature_matrix)

    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Person_ID", "Sample_ID", "Image_Path"]
        )
        writer.writeheader()
        writer.writerows(metadata)

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Total images:", len(metadata))
    print("Feature matrix shape:", feature_matrix.shape)
    print("All features finite:", np.isfinite(feature_matrix).all())
    print("Features saved:", FEATURES_FILE)
    print("Metadata saved:", METADATA_FILE)
    print("========================================")


if __name__ == "__main__":
    main()
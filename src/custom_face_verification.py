
from pathlib import Path
import csv

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    PROJECT_ROOT / "features" / "custom_face" / "custom_face_features.npy"
)
METADATA_FILE = (
    PROJECT_ROOT / "features" / "custom_face" / "custom_face_metadata.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "custom_face"
OUTPUT_FILE = OUTPUT_DIR / "custom_face_verification_results.csv"

RANDOM_STATE = 42
ENROLLMENT_COUNT = 6
VALIDATION_COUNT = 2


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-10)


def cosine_similarity(vector_a, vector_b):
    return float(np.dot(normalize(vector_a), normalize(vector_b)))


def calculate_metrics(scores, labels, threshold):
    predictions = scores >= threshold

    genuine = labels == 1
    impostor = labels == 0

    far = np.mean(predictions[impostor]) if np.any(impostor) else 0
    frr = np.mean(~predictions[genuine]) if np.any(genuine) else 0
    accuracy = np.mean(predictions == labels)

    return accuracy, far, frr


def select_threshold(scores, labels):
    unique_scores = np.sort(np.unique(scores))

    candidates = [
        unique_scores[0] - 1e-9,
        unique_scores[-1] + 1e-9,
    ]

    for i in range(len(unique_scores) - 1):
        candidates.append(
            (unique_scores[i] + unique_scores[i + 1]) / 2
        )

    best_threshold = None
    best_key = None

    for threshold in candidates:
        accuracy, far, frr = calculate_metrics(
            scores, labels, threshold
        )

        key = (abs(far - frr), (far + frr) / 2)

        if best_key is None or key < best_key:
            best_key = key
            best_threshold = threshold

    return best_threshold


def main():
    print("\n========== CUSTOM FACE VERIFICATION ==========\n")

    features = np.load(FEATURES_FILE)

    with open(METADATA_FILE, "r", newline="", encoding="utf-8") as file:
        metadata = list(csv.DictReader(file))

    if len(features) != len(metadata):
        raise ValueError("Feature and metadata counts do not match.")

    person_ids = sorted(set(row["Person_ID"] for row in metadata))

    if len(person_ids) != 2:
        raise ValueError(
            f"Expected exactly 2 participants, found {len(person_ids)}."
        )

    rng = np.random.default_rng(RANDOM_STATE)

    enrollment_indices = {}
    validation_indices = {}
    test_indices = {}

    # Split each participant independently.
    for person_id in person_ids:
        indices = np.array([
            i for i, row in enumerate(metadata)
            if row["Person_ID"] == person_id
        ])

        if len(indices) < ENROLLMENT_COUNT + VALIDATION_COUNT + 1:
            raise ValueError(
                f"Not enough samples for {person_id}."
            )

        rng.shuffle(indices)

        enrollment_indices[person_id] = indices[:ENROLLMENT_COUNT]

        validation_indices[person_id] = indices[
            ENROLLMENT_COUNT:
            ENROLLMENT_COUNT + VALIDATION_COUNT
        ]

        test_indices[person_id] = indices[
            ENROLLMENT_COUNT + VALIDATION_COUNT:
        ]

        print(
            f"{person_id}: "
            f"{len(enrollment_indices[person_id])} enrollment, "
            f"{len(validation_indices[person_id])} validation, "
            f"{len(test_indices[person_id])} test samples"
        )

    # Build one enrollment template per participant.
    templates = {}

    for person_id in person_ids:
        enrollment_features = features[
            enrollment_indices[person_id]
        ]

        normalized_features = np.array([
            normalize(vector) for vector in enrollment_features
        ])

        templates[person_id] = normalize(
            normalized_features.mean(axis=0)
        )

    # Calibrate the acceptance threshold using validation samples.
    validation_scores = []
    validation_labels = []

    for actual_id in person_ids:
        for index in validation_indices[actual_id]:
            for claimed_id in person_ids:
                score = cosine_similarity(
                    features[index],
                    templates[claimed_id]
                )

                validation_scores.append(score)
                validation_labels.append(
                    1 if actual_id == claimed_id else 0
                )

    validation_scores = np.array(validation_scores)
    validation_labels = np.array(validation_labels)

    threshold = select_threshold(
        validation_scores,
        validation_labels
    )

    print(f"\nSelected threshold: {threshold:.4f}")

    val_accuracy, val_far, val_frr = calculate_metrics(
        validation_scores,
        validation_labels,
        threshold
    )

    print("\nVALIDATION RESULTS")
    print(f"Accuracy: {val_accuracy:.2%}")
    print(f"FAR:      {val_far:.2%}")
    print(f"FRR:      {val_frr:.2%}")

    # Evaluate only on held-out test samples.
    results = []
    test_scores = []
    test_labels = []

    for actual_id in person_ids:
        for index in test_indices[actual_id]:
            for claimed_id in person_ids:
                score = cosine_similarity(
                    features[index],
                    templates[claimed_id]
                )

                genuine = actual_id == claimed_id
                accepted = score >= threshold

                test_scores.append(score)
                test_labels.append(1 if genuine else 0)

                results.append({
                    "Sample_ID": metadata[index]["Sample_ID"],
                    "Actual_ID": actual_id,
                    "Claimed_ID": claimed_id,
                    "Similarity_Score": round(score, 6),
                    "Is_Genuine": int(genuine),
                    "Accepted": int(accepted),
                    "Correct": int(accepted == genuine),
                })

    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    accuracy, far, frr = calculate_metrics(
        test_scores,
        test_labels,
        threshold
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Sample_ID",
                "Actual_ID",
                "Claimed_ID",
                "Similarity_Score",
                "Is_Genuine",
                "Accepted",
                "Correct",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n========== TEST RESULTS ==========")
    print(f"Total verification attempts: {len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"False Acceptance Rate (FAR): {far:.2%}")
    print(f"False Rejection Rate (FRR): {frr:.2%}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\n==================================")

    
if __name__ == "__main__":
    main()

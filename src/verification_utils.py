
import random
import numpy as np
import pandas as pd

from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _normalize_features(features):
    """
    Accept either:
    - One feature matrix (NumPy array).
    - Multiple feature matrices (list or tuple).

    Returns a list of feature matrices and a flag indicating
    whether the input contains only one modality.
    """

    if isinstance(features, np.ndarray):
        feature_matrices = [features]
        single_modality = True

    elif isinstance(features, (list, tuple)):
        feature_matrices = list(features)
        single_modality = False

    else:
        raise TypeError(
            "Features must be a NumPy array, list, or tuple."
        )

    if not feature_matrices:
        raise ValueError("No feature matrices were provided.")

    for matrix in feature_matrices:

        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                "Every feature matrix must be a NumPy array."
            )

        if matrix.ndim != 2:
            raise ValueError(
                "Every feature matrix must be two-dimensional."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Feature matrix contains invalid values."
            )

    row_counts = {
        matrix.shape[0]
        for matrix in feature_matrices
    }

    if len(row_counts) != 1:
        raise ValueError(
            "Feature matrices have different recording counts."
        )

    return feature_matrices, single_modality


def _format_feature_output(feature_pairs, single_modality):
    """
    Return one array for a single modality,
    or a tuple of arrays for multiple modalities.
    """

    if single_modality:
        return feature_pairs[0]

    return tuple(feature_pairs)


def _build_pair_features(feature_matrices, pairs):
    """
    Build absolute-difference feature vectors for each pair.
    Supports one or multiple biometric modalities.
    """

    pair_features = [
        []
        for _ in feature_matrices
    ]

    for idx1, idx2, _ in pairs:

        for modality_index, matrix in enumerate(feature_matrices):

            difference = np.abs(
                matrix[idx1] - matrix[idx2]
            )

            pair_features[modality_index].append(difference)

    return [
        np.asarray(values, dtype=np.float32)
        for values in pair_features
    ]


def _build_pair_info(metadata, pairs, first_name, second_name):
    """
    Create descriptive metadata for generated pairs.
    """

    pair_info = []

    for idx1, idx2, label in pairs:

        pair_info.append({
            f"{first_name}_Person":
                metadata.loc[idx1, "Person_ID"],

            f"{first_name}_Sequence":
                metadata.loc[idx1, "Sequence_ID"],

            f"{second_name}_Person":
                metadata.loc[idx2, "Person_ID"],

            f"{second_name}_Sequence":
                metadata.loc[idx2, "Sequence_ID"],

            "Label": label
        })

    return pd.DataFrame(pair_info)


# =========================================================
# SPLIT TRAIN RECORDINGS
# =========================================================

def split_train_recordings(
    metadata,
    random_state=42,
    expected_recordings=4,
    fit_recordings=3
):
    """
    Split each identity's training recordings into:
    - Fit recordings.
    - Validation recordings.

    The remaining recordings in the original metadata
    are not modified.
    """

    if not 0 < fit_recordings < expected_recordings:
        raise ValueError(
            "fit_recordings must be between 1 and expected_recordings."
        )

    rng = random.Random(random_state)

    train_indices = metadata.index[
        metadata["Split"] == "train"
    ].tolist()

    person_groups = {}

    for idx in train_indices:

        person = metadata.loc[idx, "Person_ID"]

        person_groups.setdefault(person, []).append(idx)

    fit_indices = []
    validation_indices = []

    for person in sorted(person_groups):

        indices = person_groups[person]

        if len(indices) != expected_recordings:
            raise ValueError(
                f"Expected {expected_recordings} training recordings "
                f"for {person}, found {len(indices)}."
            )

        shuffled = indices.copy()
        rng.shuffle(shuffled)

        fit_indices.extend(
            shuffled[:fit_recordings]
        )

        validation_indices.extend(
            shuffled[fit_recordings:]
        )

    print("\n========== INTERNAL SPLIT ==========")
    print("Fit recordings:", len(fit_indices))
    print("Validation recordings:", len(validation_indices))
    print("Unique identities:", len(person_groups))
    print("====================================")

    return fit_indices, validation_indices


# =========================================================
# CREATE TRAINING PAIRS
# =========================================================

def create_training_pairs(
    features,
    metadata,
    fit_indices,
    impostor_ratio=1,
    random_state=42
):
    """
    Generate genuine and impostor training pairs.

    impostor_ratio:
        1  = one impostor pair per genuine pair.
        10 = ten impostor pairs per genuine pair.

    Returns:
        X_pairs, y_pairs

    For multiple modalities, X_pairs is a tuple containing
    one pair-feature matrix per modality.
    """

    if impostor_ratio <= 0:
        raise ValueError(
            "impostor_ratio must be greater than zero."
        )

    feature_matrices, single_modality = (
        _normalize_features(features)
    )

    rng = random.Random(random_state)

    person_groups = {}

    for idx in fit_indices:

        person = metadata.loc[idx, "Person_ID"]

        person_groups.setdefault(person, []).append(idx)

    genuine_pairs = []

    for person in sorted(person_groups):

        indices = sorted(person_groups[person])

        for idx1, idx2 in combinations(indices, 2):

            genuine_pairs.append(
                (idx1, idx2, 1)
            )

    target_count = int(
        len(genuine_pairs) * impostor_ratio
    )

    if target_count < 1:
        raise ValueError(
            "Not enough genuine pairs to generate impostor pairs."
        )

    impostor_pairs = set()

    attempts = 0
    max_attempts = target_count * 100

    while (
        len(impostor_pairs) < target_count
        and attempts < max_attempts
    ):

        idx1, idx2 = rng.sample(
            fit_indices,
            2
        )

        person1 = metadata.loc[idx1, "Person_ID"]
        person2 = metadata.loc[idx2, "Person_ID"]

        if person1 != person2:

            pair = tuple(
                sorted((idx1, idx2))
            )

            impostor_pairs.add(pair)

        attempts += 1

    if len(impostor_pairs) < target_count:
        raise RuntimeError(
            "Could not generate enough impostor pairs."
        )

    # Sorting makes selection reproducible.
    selected_impostors = sorted(impostor_pairs)[
        :target_count
    ]

    all_pairs = genuine_pairs + [
        (idx1, idx2, 0)
        for idx1, idx2 in selected_impostors
    ]

    rng.shuffle(all_pairs)

    pair_features = _build_pair_features(
        feature_matrices,
        all_pairs
    )

    y_pairs = np.asarray(
        [label for _, _, label in all_pairs],
        dtype=np.int32
    )

    print("\n========== TRAINING PAIRS ==========")
    print("Genuine pairs:", len(genuine_pairs))
    print("Impostor pairs:", len(selected_impostors))
    print("Total pairs:", len(all_pairs))

    for index, matrix in enumerate(pair_features):
        print(
            f"Modality {index + 1} pair shape:",
            matrix.shape
        )

    print("====================================")

    return (
        _format_feature_output(
            pair_features,
            single_modality
        ),
        y_pairs
    )


# =========================================================
# CREATE VALIDATION PAIRS
# =========================================================

def create_validation_pairs(
    features,
    metadata,
    fit_indices,
    validation_indices
):
    """
    Compare every validation recording against every fit recording.

    Returns:
        X_validation, y_validation, pair_info
    """

    feature_matrices, single_modality = (
        _normalize_features(features)
    )

    pairs = []

    for validation_idx in validation_indices:

        validation_person = metadata.loc[
            validation_idx, "Person_ID"
        ]

        for fit_idx in fit_indices:

            fit_person = metadata.loc[
                fit_idx, "Person_ID"
            ]

            label = int(
                validation_person == fit_person
            )

            pairs.append(
                (validation_idx, fit_idx, label)
            )

    pair_features = _build_pair_features(
        feature_matrices,
        pairs
    )

    y_validation = np.asarray(
        [label for _, _, label in pairs],
        dtype=np.int32
    )

    pair_info = _build_pair_info(
        metadata,
        pairs,
        "Validation",
        "Fit"
    )

    print("\n========== VALIDATION PAIRS ==========")
    print("Total pairs:", len(y_validation))
    print("Genuine pairs:", int(np.sum(y_validation == 1)))
    print("Impostor pairs:", int(np.sum(y_validation == 0)))
    print("======================================")

    return (
        _format_feature_output(
            pair_features,
            single_modality
        ),
        y_validation,
        pair_info
    )


# =========================================================
# CREATE TEST PAIRS
# =========================================================

def create_test_pairs(features, metadata):
    """
    Compare every test recording against every enrollment recording.

    Returns:
        X_test, y_test, pair_info
    """

    feature_matrices, single_modality = (
        _normalize_features(features)
    )

    enrollment_indices = metadata.index[
        metadata["Split"] == "enrollment"
    ].tolist()

    test_indices = metadata.index[
        metadata["Split"] == "test"
    ].tolist()

    pairs = []

    for test_idx in test_indices:

        test_person = metadata.loc[
            test_idx, "Person_ID"
        ]

        for enrollment_idx in enrollment_indices:

            enrollment_person = metadata.loc[
                enrollment_idx, "Person_ID"
            ]

            label = int(
                test_person == enrollment_person
            )

            pairs.append(
                (test_idx, enrollment_idx, label)
            )

    pair_features = _build_pair_features(
        feature_matrices,
        pairs
    )

    y_test = np.asarray(
        [label for _, _, label in pairs],
        dtype=np.int32
    )

    pair_info = _build_pair_info(
        metadata,
        pairs,
        "Test",
        "Enrollment"
    )

    print("\n========== TEST PAIRS ==========")
    print("Total pairs:", len(y_test))
    print("Genuine pairs:", int(np.sum(y_test == 1)))
    print("Impostor pairs:", int(np.sum(y_test == 0)))
    print("================================")

    return (
        _format_feature_output(
            pair_features,
            single_modality
        ),
        y_test,
        pair_info
    )


# =========================================================
# SELECT THRESHOLD
# =========================================================

def select_threshold(
    y_validation,
    validation_scores
):
    """
    Select a threshold using validation data only.

    The selected threshold minimizes the absolute
    difference between FAR and FRR.
    """

    fpr, tpr, thresholds = roc_curve(
        y_validation,
        validation_scores,
        pos_label=1
    )

    fnr = 1 - tpr

    best_index = int(
        np.argmin(np.abs(fpr - fnr))
    )

    threshold = float(
        thresholds[best_index]
    )

    validation_far = fpr[best_index]
    validation_frr = fnr[best_index]

    print("\n========== THRESHOLD SELECTION ==========")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Validation FAR: {validation_far:.4f}")
    print(f"Validation FRR: {validation_frr:.4f}")
    print("=========================================")

    return threshold


# =========================================================
# CALCULATE VERIFICATION METRICS
# =========================================================

def calculate_metrics(y_true, predictions):
    """
    Calculate standard biometric verification metrics.

    Class convention:
        0 = Impostor
        1 = Genuine

    Confusion matrix layout:
        [[TN, FP],
         [FN, TP]]
    """

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    far = (
        fp / (fp + tn)
        if (fp + tn)
        else 0
    )

    frr = (
        fn / (fn + tp)
        if (fn + tp)
        else 0
    )

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "FAR": far,
        "FRR": frr,

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


import os
import random
import joblib
import numpy as np
import pandas as pd

from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
IMPOSTOR_RATIO = 10

VOICE_COLUMNS = (
    list(range(0, 26)) + [78, 79, 80, 81]
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FACE_FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "features", "face", "face_features.npy"
)

FACE_METADATA_PATH = os.path.join(
    PROJECT_ROOT, "features", "face", "face_metadata.csv"
)

VOICE_FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "features", "voice", "voice_features.npy"
)

VOICE_METADATA_PATH = os.path.join(
    PROJECT_ROOT, "features", "voice", "voice_metadata.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT, "models", "hierarchical"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT, "results", "hierarchical"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

def load_data():

    face_features = np.load(FACE_FEATURES_PATH)
    voice_features = np.load(VOICE_FEATURES_PATH)

    face_metadata = pd.read_csv(FACE_METADATA_PATH)
    voice_metadata = pd.read_csv(VOICE_METADATA_PATH)

    voice_features = voice_features[:, VOICE_COLUMNS]

    required_columns = [
        "Person_ID",
        "Sequence_ID",
        "Split"
    ]

    for column in required_columns:

        if column not in face_metadata.columns:
            raise ValueError(
                f"Missing column in face metadata: {column}"
            )

        if column not in voice_metadata.columns:
            raise ValueError(
                f"Missing column in voice metadata: {column}"
            )

    if not face_metadata[required_columns].equals(
        voice_metadata[required_columns]
    ):
        raise ValueError(
            "Face and Voice metadata are not aligned."
        )

    if len(face_features) != len(face_metadata):
        raise ValueError("Face feature/metadata mismatch.")

    if len(voice_features) != len(voice_metadata):
        raise ValueError("Voice feature/metadata mismatch.")

    if not np.isfinite(face_features).all():
        raise ValueError("Face features contain invalid values.")

    if not np.isfinite(voice_features).all():
        raise ValueError("Voice features contain invalid values.")

    if len(face_features) != 430:
        raise ValueError(
            f"Expected 430 recordings, found {len(face_features)}"
        )

    print("\n========== DATA VALIDATION ==========")
    print("Face shape:", face_features.shape)
    print("Voice shape:", voice_features.shape)
    print("Total recordings:", len(face_metadata))
    print("Unique identities:",
          face_metadata["Person_ID"].nunique())
    print("Split counts:")
    print(face_metadata["Split"].value_counts().to_dict())
    print("Metadata alignment: PASSED")
    print("Finite features: PASSED")
    print("=====================================")

    return face_features, voice_features, face_metadata


# =========================================================
# INTERNAL FIT / VALIDATION SPLIT
# =========================================================

def split_train_recordings(metadata):

    rng = random.Random(RANDOM_STATE)

    train_indices = metadata.index[
        metadata["Split"] == "train"
    ].tolist()

    person_groups = {}

    for idx in train_indices:

        person = metadata.loc[idx, "Person_ID"]

        person_groups.setdefault(person, []).append(idx)

    fit_indices = []
    validation_indices = []

    for person, indices in person_groups.items():

        if len(indices) != 4:
            raise ValueError(
                f"Expected 4 train recordings for {person}, "
                f"found {len(indices)}"
            )

        shuffled = indices.copy()
        rng.shuffle(shuffled)

        fit_indices.extend(shuffled[:3])
        validation_indices.append(shuffled[3])

    print("\n========== INTERNAL SPLIT ==========")
    print("Fit recordings:", len(fit_indices))
    print("Validation recordings:", len(validation_indices))
    print("====================================")

    return fit_indices, validation_indices


# =========================================================
# CREATE TRAINING PAIRS
# =========================================================

def create_training_pairs(
    face_features,
    voice_features,
    metadata,
    fit_indices
):

    rng = random.Random(RANDOM_STATE)

    person_groups = {}

    for idx in fit_indices:

        person = metadata.loc[idx, "Person_ID"]

        person_groups.setdefault(person, []).append(idx)

    genuine_pairs = []

    for person, indices in person_groups.items():

        for idx1, idx2 in combinations(indices, 2):
            genuine_pairs.append((idx1, idx2, 1))

    impostor_pairs = set()

    target_count = len(genuine_pairs)
    attempts = 0
    max_attempts = target_count * 100

    while (
        len(impostor_pairs) < target_count * IMPOSTOR_RATIO
        and attempts < max_attempts
    ):

        idx1, idx2 = rng.sample(fit_indices, 2)

        person1 = metadata.loc[idx1, "Person_ID"]
        person2 = metadata.loc[idx2, "Person_ID"]

        if person1 != person2:

            pair = tuple(sorted((idx1, idx2)))
            impostor_pairs.add(pair)

        attempts += 1

    required_impostors = target_count * IMPOSTOR_RATIO

    if len(impostor_pairs) < required_impostors:
        raise RuntimeError(
            "Could not generate enough impostor pairs."
        )

    impostor_pairs = [
        (idx1, idx2, 0)
        for idx1, idx2 in list(impostor_pairs)[
            :required_impostors
        ]
    ]

    all_pairs = genuine_pairs + impostor_pairs
    rng.shuffle(all_pairs)

    X_face = []
    X_voice = []
    y = []

    for idx1, idx2, label in all_pairs:

        X_face.append(
            np.abs(
                face_features[idx1] - face_features[idx2]
            )
        )

        X_voice.append(
            np.abs(
                voice_features[idx1] - voice_features[idx2]
            )
        )

        y.append(label)

    X_face = np.asarray(X_face, dtype=np.float32)
    X_voice = np.asarray(X_voice, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print("\n========== TRAINING PAIRS ==========")
    print("Genuine:", int(np.sum(y == 1)))
    print("Impostor:", int(np.sum(y == 0)))
    print("Total:", len(y))
    print("Face pair shape:", X_face.shape)
    print("Voice pair shape:", X_voice.shape)
    print("====================================")

    return X_face, X_voice, y


# =========================================================
# CREATE VALIDATION PAIRS
# =========================================================

def create_validation_pairs(
    face_features,
    voice_features,
    metadata,
    fit_indices,
    validation_indices
):

    X_face = []
    X_voice = []
    y = []

    for validation_idx in validation_indices:

        validation_person = metadata.loc[
            validation_idx, "Person_ID"
        ]

        for fit_idx in fit_indices:

            fit_person = metadata.loc[
                fit_idx, "Person_ID"
            ]

            label = int(validation_person == fit_person)

            X_face.append(
                np.abs(
                    face_features[validation_idx]
                    - face_features[fit_idx]
                )
            )

            X_voice.append(
                np.abs(
                    voice_features[validation_idx]
                    - voice_features[fit_idx]
                )
            )

            y.append(label)

    X_face = np.asarray(X_face, dtype=np.float32)
    X_voice = np.asarray(X_voice, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print("\n========== VALIDATION PAIRS ==========")
    print("Total:", len(y))
    print("Genuine:", int(np.sum(y == 1)))
    print("Impostor:", int(np.sum(y == 0)))
    print("======================================")

    return X_face, X_voice, y


# =========================================================
# CREATE FINAL TEST PAIRS
# =========================================================

def create_test_pairs(
    face_features,
    voice_features,
    metadata
):

    enrollment_indices = metadata.index[
        metadata["Split"] == "enrollment"
    ].tolist()

    test_indices = metadata.index[
        metadata["Split"] == "test"
    ].tolist()

    X_face = []
    X_voice = []
    y = []
    pair_info = []

    for test_idx in test_indices:

        test_person = metadata.loc[
            test_idx, "Person_ID"
        ]

        for enrollment_idx in enrollment_indices:

            enrollment_person = metadata.loc[
                enrollment_idx, "Person_ID"
            ]

            label = int(test_person == enrollment_person)

            X_face.append(
                np.abs(
                    face_features[test_idx]
                    - face_features[enrollment_idx]
                )
            )

            X_voice.append(
                np.abs(
                    voice_features[test_idx]
                    - voice_features[enrollment_idx]
                )
            )

            y.append(label)

            pair_info.append({
                "Test_Person": test_person,
                "Test_Sequence": metadata.loc[
                    test_idx, "Sequence_ID"
                ],
                "Enrollment_Person": enrollment_person,
                "Enrollment_Sequence": metadata.loc[
                    enrollment_idx, "Sequence_ID"
                ],
                "Label": label
            })

    X_face = np.asarray(X_face, dtype=np.float32)
    X_voice = np.asarray(X_voice, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    pair_info = pd.DataFrame(pair_info)

    print("\n========== TEST PAIRS ==========")
    print("Total:", len(y))
    print("Genuine:", int(np.sum(y == 1)))
    print("Impostor:", int(np.sum(y == 0)))
    print("================================")

    return X_face, X_voice, y, pair_info


# =========================================================
# SELECT VOICE THRESHOLD USING VALIDATION ONLY
# =========================================================

def select_voice_threshold(y_validation, voice_scores):

    fpr, tpr, thresholds = roc_curve(
        y_validation,
        voice_scores,
        pos_label=1
    )

    fnr = 1 - tpr

    best_index = np.argmin(np.abs(fpr - fnr))

    threshold = float(thresholds[best_index])

    print("\n========== VOICE THRESHOLD ==========")
    print(f"Threshold: {threshold:.6f}")
    print(f"Validation FAR: {fpr[best_index]:.4f}")
    print(f"Validation FRR: {fnr[best_index]:.4f}")
    print("=====================================")

    return threshold


# =========================================================
# SELECT FACE CONFIDENCE THRESHOLD
# =========================================================

def select_face_confidence_threshold(
    y_validation,
    face_scores,
    voice_scores,
    voice_threshold
):

    face_predictions = (face_scores >= 0.5).astype(int)

    face_confidence = np.maximum(
        face_scores,
        1 - face_scores
    )

    voice_predictions = (
        voice_scores >= voice_threshold
    ).astype(int)

    candidates = np.unique(
        np.concatenate([
            np.array([0.5, 1.0]),
            face_confidence
        ])
    )

    best_threshold = 0.5
    best_balanced_accuracy = -1
    best_fallback_count = len(y_validation)

    for threshold in candidates:

        use_voice = face_confidence < threshold

        final_predictions = np.where(
            use_voice,
            voice_predictions,
            face_predictions
        )

        score = balanced_accuracy_score(
            y_validation,
            final_predictions
        )

        fallback_count = int(np.sum(use_voice))

        if (
            score > best_balanced_accuracy
            or (
                np.isclose(score, best_balanced_accuracy)
                and fallback_count < best_fallback_count
            )
        ):

            best_balanced_accuracy = score
            best_threshold = float(threshold)
            best_fallback_count = fallback_count

    print("\n========== FACE CONFIDENCE THRESHOLD ==========")
    print(f"Selected threshold: {best_threshold:.6f}")
    print(f"Validation balanced accuracy: "
          f"{best_balanced_accuracy:.4f}")
    print(f"Validation Voice fallbacks: {best_fallback_count}")
    print("================================================")

    return best_threshold


# =========================================================
# EVALUATION
# =========================================================

def calculate_metrics(y_true, predictions):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) else 0
    frr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "FAR": far,
        "FRR": frr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# =========================================================
# MAIN
# =========================================================

def main():

    face_features, voice_features, metadata = load_data()

    fit_indices, validation_indices = (
        split_train_recordings(metadata)
    )

    (
        X_face_train,
        X_voice_train,
        y_train
    ) = create_training_pairs(
        face_features,
        voice_features,
        metadata,
        fit_indices
    )

    (
        X_face_validation,
        X_voice_validation,
        y_validation
    ) = create_validation_pairs(
        face_features,
        voice_features,
        metadata,
        fit_indices,
        validation_indices
    )

    # -----------------------------------------------------
    # TRAIN FACE RANDOM FOREST
    # -----------------------------------------------------

    print("\n========== TRAIN FACE MODEL ==========")

    face_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    face_model.fit(X_face_train, y_train)

    face_validation_scores = face_model.predict_proba(
        X_face_validation
    )[:, 1]

    # -----------------------------------------------------
    # TRAIN VOICE SVM
    # -----------------------------------------------------

    print("\n========== TRAIN VOICE MODEL ==========")

    voice_scaler = StandardScaler()

    X_voice_train_scaled = voice_scaler.fit_transform(
        X_voice_train
    )

    X_voice_validation_scaled = voice_scaler.transform(
        X_voice_validation
    )

    voice_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    voice_model.fit(
        X_voice_train_scaled,
        y_train
    )

    voice_validation_scores = voice_model.decision_function(
        X_voice_validation_scaled
    )

    # -----------------------------------------------------
    # VALIDATION THRESHOLDS
    # -----------------------------------------------------

    voice_threshold = select_voice_threshold(
        y_validation,
        voice_validation_scores
    )

    face_confidence_threshold = (
        select_face_confidence_threshold(
            y_validation,
            face_validation_scores,
            voice_validation_scores,
            voice_threshold
        )
    )

    # -----------------------------------------------------
    # SAVE MODELS AND CONFIGURATION
    # -----------------------------------------------------

    joblib.dump(
        face_model,
        os.path.join(
            MODEL_DIR, "hierarchical_face_random_forest.pkl"
        )
    )

    joblib.dump(
        {
            "model": voice_model,
            "scaler": voice_scaler,
            "threshold": voice_threshold,
            "selected_columns": VOICE_COLUMNS
        },
        os.path.join(
            MODEL_DIR, "hierarchical_voice_svm.pkl"
        )
    )

    joblib.dump(
        {
            "face_confidence_threshold":
                face_confidence_threshold,
            "voice_threshold": voice_threshold,
            "face_model": "RandomForest",
            "voice_model": "SVM",
            "voice_features": "static_mfcc_rms_zcr"
        },
        os.path.join(
            MODEL_DIR, "hierarchical_configuration.pkl"
        )
    )

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    (
        X_face_test,
        X_voice_test,
        y_test,
        pair_info
    ) = create_test_pairs(
        face_features,
        voice_features,
        metadata
    )

    face_test_scores = face_model.predict_proba(
        X_face_test
    )[:, 1]

    face_test_predictions = (
        face_test_scores >= 0.5
    ).astype(int)

    face_test_confidence = np.maximum(
        face_test_scores,
        1 - face_test_scores
    )

    X_voice_test_scaled = voice_scaler.transform(
        X_voice_test
    )

    voice_test_scores = voice_model.decision_function(
        X_voice_test_scaled
    )

    voice_test_predictions = (
        voice_test_scores >= voice_threshold
    ).astype(int)

    use_voice = (
        face_test_confidence < face_confidence_threshold
    )

    final_predictions = np.where(
        use_voice,
        voice_test_predictions,
        face_test_predictions
    )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    face_metrics = calculate_metrics(
        y_test,
        face_test_predictions
    )

    voice_metrics = calculate_metrics(
        y_test,
        voice_test_predictions
    )

    hierarchical_metrics = calculate_metrics(
        y_test,
        final_predictions
    )

    print("\n========== TEST RESULTS ==========")

    for model_name, metrics in [
        ("Face RF", face_metrics),
        ("Voice SVM", voice_metrics),
        ("Hierarchical", hierarchical_metrics)
    ]:

        print(f"\n{model_name}")

        for metric, value in metrics.items():

            if metric in ["TN", "FP", "FN", "TP"]:
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")

    print("\n========== CASCADE USAGE ==========")
    print("Total test pairs:", len(y_test))
    print("Voice fallbacks:", int(np.sum(use_voice)))
    print("Face-only decisions:", int(np.sum(~use_voice)))
    print("Voice fallback rate:",
          f"{np.mean(use_voice):.4f}")
    print("==================================")

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    pair_info["Face_Score"] = face_test_scores
    pair_info["Face_Confidence"] = face_test_confidence
    pair_info["Face_Prediction"] = face_test_predictions

    pair_info["Voice_Score"] = voice_test_scores
    pair_info["Voice_Prediction"] = voice_test_predictions

    pair_info["Used_Voice"] = use_voice.astype(int)
    pair_info["Final_Prediction"] = final_predictions

    pair_info.to_csv(
        os.path.join(
            RESULT_DIR, "hierarchical_test_predictions.csv"
        ),
        index=False
    )

    metrics_table = pd.DataFrame([
        {"Model": "Face RF", **face_metrics},
        {"Model": "Voice SVM", **voice_metrics},
        {"Model": "Hierarchical", **hierarchical_metrics}
    ])

    metrics_table.to_csv(
        os.path.join(
            RESULT_DIR, "hierarchical_model_comparison.csv"
        ),
        index=False
    )

    np.save(
        os.path.join(
            RESULT_DIR, "hierarchical_confusion_matrix.npy"
        ),
        confusion_matrix(
            y_test,
            final_predictions,
            labels=[0, 1]
        )
    )

    print("\nResults saved to:", RESULT_DIR)
    print("Models saved to:", MODEL_DIR)
    print("\nHIERARCHICAL VERIFICATION COMPLETED.")


if __name__ == "__main__":
    main()

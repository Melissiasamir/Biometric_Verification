
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

try:
    from .verification_utils import (
        split_train_recordings,
        create_training_pairs,
        create_validation_pairs,
        create_test_pairs,
        select_threshold,
        calculate_metrics
    )
except ImportError:
    from verification_utils import (
        split_train_recordings,
        create_training_pairs,
        create_validation_pairs,
        create_test_pairs,
        select_threshold,
        calculate_metrics
    )


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
IMPOSTOR_RATIO = 10

FEATURE_MODE = "early_fusion_face_voice"
EXPECTED_FEATURE_COUNT = 3578

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURES_PATH = os.path.join(
    PROJECT_ROOT,
    "features",
    "fusion",
    "fusion_features.npy"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "features",
    "fusion",
    "fusion_metadata.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "fusion"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "fusion"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_PREFIX = "fusion"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    features = np.load(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if features.ndim != 2:
        raise ValueError(
            "Fusion feature matrix must be two-dimensional."
        )

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} fusion features, "
            f"got {features.shape[1]}"
        )

    if len(features) != len(metadata):
        raise ValueError(
            "Feature and metadata counts do not match."
        )

    required_columns = [
        "Person_ID",
        "Sequence_ID",
        "Split"
    ]

    if not all(
        column in metadata.columns
        for column in required_columns
    ):
        raise ValueError(
            "Fusion metadata is missing required columns."
        )

    if metadata[required_columns].isna().any().any():
        raise ValueError(
            "Fusion metadata contains missing values."
        )

    if metadata.duplicated(
        subset=["Person_ID", "Sequence_ID"]
    ).any():
        raise ValueError(
            "Duplicate recordings found in metadata."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Fusion feature matrix contains invalid values."
        )

    expected_splits = {
        "train": 172,
        "enrollment": 86,
        "test": 172
    }

    actual_splits = metadata["Split"].value_counts().to_dict()

    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected split counts: {actual_splits}"
        )

    if metadata["Person_ID"].nunique() != 43:
        raise ValueError(
            "Expected 43 unique identities."
        )

    print("\n========== FUSION DATA ==========")
    print("Feature mode:", FEATURE_MODE)
    print("Feature shape:", features.shape)
    print("Metadata records:", len(metadata))
    print("\nSplit counts:")
    print(metadata["Split"].value_counts())
    print("Unique identities:", metadata["Person_ID"].nunique())
    print("All features finite:", np.isfinite(features).all())
    print("================================")

    return features, metadata


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    y_true,
    scores,
    threshold,
    model_name
):

    predictions = (
        scores >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_true,
        predictions
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    print(f"\n========== {model_name} RESULTS ==========")
    print(f"Threshold:          {threshold:.6f}")
    print(f"Accuracy:           {metrics['Accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['Balanced_Accuracy']:.4f}")
    print(f"Precision:          {metrics['Precision']:.4f}")
    print(f"Recall:             {metrics['Recall']:.4f}")
    print(f"F1-score:           {metrics['F1']:.4f}")
    print(f"FAR:                {metrics['FAR']:.4f}")
    print(f"FRR:                {metrics['FRR']:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Impostor", "Genuine"],
            zero_division=0
        )
    )

    results = {
        "Feature_Mode": FEATURE_MODE,
        "Feature_Count": EXPECTED_FEATURE_COUNT,
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": metrics["Accuracy"],
        "Balanced_Accuracy": metrics["Balanced_Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1_Score": metrics["F1"],
        "FAR": metrics["FAR"],
        "FRR": metrics["FRR"],
        "TN": metrics["TN"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "TP": metrics["TP"]
    }

    return results, predictions, cm


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    np.random.seed(RANDOM_STATE)

    features, metadata = load_data()

    # Split the four training recordings per person:
    # 3 for fitting and 1 for validation.
    fit_indices, validation_indices = split_train_recordings(
        metadata,
        random_state=RANDOM_STATE,
        expected_recordings=4,
        fit_recordings=3
    )

    # Generate training pairs with a 1:10 genuine/impostor ratio.
    X_train, y_train = create_training_pairs(
        features,
        metadata,
        fit_indices,
        impostor_ratio=IMPOSTOR_RATIO,
        random_state=RANDOM_STATE
    )

    # Generate validation pairs.
    X_validation, y_validation, validation_info = (
        create_validation_pairs(
            features,
            metadata,
            fit_indices,
            validation_indices
        )
    )

    # Generate final test pairs using enrollment recordings.
    X_test, y_test, test_info = create_test_pairs(
        features,
        metadata
    )

    # =============================================
    # SVM SCALING
    # Fit the scaler on training pairs only.
    # =============================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # =============================================
    # MODEL 1: SVM
    # =============================================

    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    print("\nTraining SVM...")

    svm_model.fit(
        X_train_scaled,
        y_train
    )

    svm_validation_scores = svm_model.decision_function(
        X_validation_scaled
    )

    svm_threshold = select_threshold(
        y_validation,
        svm_validation_scores
    )

    svm_test_scores = svm_model.decision_function(
        X_test_scaled
    )

    svm_results, svm_predictions, svm_cm = evaluate_model(
        y_test,
        svm_test_scores,
        svm_threshold,
        "SVM"
    )

    joblib.dump(
        {
            "model": svm_model,
            "scaler": scaler,
            "threshold": svm_threshold,
            "feature_mode": FEATURE_MODE,
            "feature_count": EXPECTED_FEATURE_COUNT
        },
        os.path.join(
            MODEL_DIR,
            f"{OUTPUT_PREFIX}_svm.pkl"
        )
    )

    # =============================================
    # MODEL 2: RANDOM FOREST
    # No scaling is applied to Random Forest.
    # =============================================

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    rf_model.fit(
        X_train,
        y_train
    )

    rf_validation_scores = rf_model.predict_proba(
        X_validation
    )[:, 1]

    rf_threshold = select_threshold(
        y_validation,
        rf_validation_scores
    )

    rf_test_scores = rf_model.predict_proba(
        X_test
    )[:, 1]

    rf_results, rf_predictions, rf_cm = evaluate_model(
        y_test,
        rf_test_scores,
        rf_threshold,
        "Random Forest"
    )

    joblib.dump(
        {
            "model": rf_model,
            "threshold": rf_threshold,
            "feature_mode": FEATURE_MODE,
            "feature_count": EXPECTED_FEATURE_COUNT
        },
        os.path.join(
            MODEL_DIR,
            f"{OUTPUT_PREFIX}_random_forest.pkl"
        )
    )

    # =============================================
    # SAVE MODEL COMPARISON
    # =============================================

    results_df = pd.DataFrame([
        svm_results,
        rf_results
    ])

    results_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"{OUTPUT_PREFIX}_model_comparison.csv"
        ),
        index=False
    )

    # =============================================
    # SAVE TEST PREDICTIONS
    # =============================================

    test_info["SVM_Score"] = svm_test_scores
    test_info["SVM_Prediction"] = svm_predictions

    test_info["RF_Score"] = rf_test_scores
    test_info["RF_Prediction"] = rf_predictions

    test_info.to_csv(
        os.path.join(
            RESULT_DIR,
            f"{OUTPUT_PREFIX}_test_predictions.csv"
        ),
        index=False
    )

    # =============================================
    # SAVE VALIDATION PREDICTIONS
    # =============================================

    validation_info["SVM_Score"] = svm_validation_scores
    validation_info["RF_Score"] = rf_validation_scores

    validation_info.to_csv(
        os.path.join(
            RESULT_DIR,
            f"{OUTPUT_PREFIX}_validation_predictions.csv"
        ),
        index=False
    )

    # =============================================
    # SAVE CONFUSION MATRICES
    # =============================================

    np.save(
        os.path.join(
            RESULT_DIR,
            f"{OUTPUT_PREFIX}_svm_confusion_matrix.npy"
        ),
        svm_cm
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            f"{OUTPUT_PREFIX}_rf_confusion_matrix.npy"
        ),
        rf_cm
    )

    # =============================================
    # COMPLETION REPORT
    # =============================================

    print("\n========== FUSION VERIFICATION COMPLETE ==========")
    print("Feature mode:", FEATURE_MODE)
    print("Feature count:", EXPECTED_FEATURE_COUNT)
    print("Models saved in:", MODEL_DIR)
    print("Results saved in:", RESULT_DIR)
    print("==================================================")


if __name__ == "__main__":
    main()


import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

# Current experiment: Static MFCC + RMS + ZCR
FEATURE_MODE = "static_mfcc_rms_zcr"

FEATURE_COLUMNS = {
    "static_mfcc": list(range(0, 26)),

    "static_mfcc_rms": (
        list(range(0, 26)) + [80, 81]
    ),

    "static_mfcc_rms_zcr": (
        list(range(0, 26)) + [78, 79, 80, 81]
    ),

    "all_features": list(range(82))
}

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "features", "voice", "voice_features.npy"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT, "features", "voice", "voice_metadata.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "voice")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "voice")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_PREFIX = f"voice_{FEATURE_MODE}"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    features = np.load(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if FEATURE_MODE not in FEATURE_COLUMNS:
        raise ValueError(
            f"Unknown feature mode: {FEATURE_MODE}"
        )

    if len(features) != len(metadata):
        raise ValueError(
            "Feature and metadata counts do not match."
        )

    if features.shape[1] != 82:
        raise ValueError(
            f"Expected 82 voice features, got {features.shape[1]}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature matrix contains invalid values."
        )

    if metadata.duplicated(
        subset=["Person_ID", "Sequence_ID"]
    ).any():
        raise ValueError(
            "Duplicate recordings found in metadata."
        )

    selected_columns = FEATURE_COLUMNS[FEATURE_MODE]
    features = features[:, selected_columns]

    print("\n========== VOICE DATA ==========")
    print("Feature mode:", FEATURE_MODE)
    print("Selected feature count:", features.shape[1])
    print("Feature shape:", features.shape)
    print("Metadata records:", len(metadata))
    print(metadata["Split"].value_counts())
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

    accuracy = metrics["Accuracy"]
    balanced_accuracy = metrics["Balanced_Accuracy"]
    precision = metrics["Precision"]
    recall = metrics["Recall"]
    f1 = metrics["F1"]
    far = metrics["FAR"]
    frr = metrics["FRR"]

    tn = metrics["TN"]
    fp = metrics["FP"]
    fn = metrics["FN"]
    tp = metrics["TP"]

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    print(f"\n========== {model_name} RESULTS ==========")
    print(f"Threshold:          {threshold:.6f}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1-score:           {f1:.4f}")
    print(f"FAR:                {far:.4f}")
    print(f"FRR:                {frr:.4f}")

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
        "Feature_Count": len(FEATURE_COLUMNS[FEATURE_MODE]),
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "FAR": far,
        "FRR": frr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    return results, predictions, cm


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    np.random.seed(RANDOM_STATE)

    features, metadata = load_data()

    # =============================================
    # FIT / VALIDATION SPLIT
    # =============================================

    fit_indices, validation_indices = (
        split_train_recordings(
            metadata,
            random_state=RANDOM_STATE,
            expected_recordings=4,
            fit_recordings=3
        )
    )

    # =============================================
    # CREATE TRAINING PAIRS
    # =============================================

    X_train, y_train = create_training_pairs(
        features,
        metadata,
        fit_indices,
        impostor_ratio=IMPOSTOR_RATIO,
        random_state=RANDOM_STATE
    )

    # =============================================
    # CREATE VALIDATION PAIRS
    # =============================================

    X_validation, y_validation, validation_info = (
        create_validation_pairs(
            features,
            metadata,
            fit_indices,
            validation_indices
        )
    )

    # =============================================
    # CREATE TEST PAIRS
    # =============================================

    X_test, y_test, test_info = create_test_pairs(
        features,
        metadata
    )

    # =============================================
    # SCALE USING TRAINING PAIRS ONLY
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
            "selected_columns": FEATURE_COLUMNS[FEATURE_MODE]
        },
        os.path.join(
            MODEL_DIR,
            f"{OUTPUT_PREFIX}_svm.pkl"
        )
    )

    # =============================================
    # MODEL 2: RANDOM FOREST
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
            "selected_columns": FEATURE_COLUMNS[FEATURE_MODE]
        },
        os.path.join(
            MODEL_DIR,
            f"{OUTPUT_PREFIX}_random_forest.pkl"
        )
    )

    # =============================================
    # SAVE RESULTS
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

    print("\n========== VOICE VERIFICATION COMPLETE ==========")
    print("Feature mode:", FEATURE_MODE)
    print("Selected features:", len(FEATURE_COLUMNS[FEATURE_MODE]))
    print("Models saved in:", MODEL_DIR)
    print("Results saved in:", RESULT_DIR)
    print("==================================================")


if __name__ == "__main__":
    main()

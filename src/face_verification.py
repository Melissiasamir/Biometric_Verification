
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

try:
    from .verification_utils import (
        create_training_pairs,
        create_test_pairs,
        calculate_metrics
    )
except ImportError:
    from verification_utils import (
        create_training_pairs,
        create_test_pairs,
        calculate_metrics
    )


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "features", "face", "face_features.npy"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT, "features", "face", "face_metadata.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "face")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "face")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    features = np.load(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if len(features) != len(metadata):
        raise ValueError(
            "Feature and metadata counts do not match."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature matrix contains invalid values."
        )

    print("\n========== FACE DATA ==========")
    print("Feature shape:", features.shape)
    print("Metadata records:", len(metadata))
    print("Split counts:")
    print(metadata["Split"].value_counts())
    print("================================")

    return features, metadata


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(f"\n========== {model_name} RESULTS ==========")
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
            y_test,
            predictions,
            target_names=["Impostor", "Genuine"],
            zero_division=0
        )
    )

    results = {
        "Model": model_name,
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

    # Use ALL four training recordings per person.
    # No internal validation split for Face verification.
    train_indices = metadata.index[
        metadata["Split"] == "train"
    ].tolist()

    # Balanced genuine and impostor training pairs (1:1).
    X_train, y_train = create_training_pairs(
        features,
        metadata,
        train_indices,
        impostor_ratio=1,
        random_state=RANDOM_STATE
    )

    # Compare every test recording against all enrollment recordings.
    X_test, y_test, pair_info = create_test_pairs(
        features,
        metadata
    )

    # Fit scaler on training pairs only.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------
    # MODEL 1: SVM
    # ---------------------------------------------

    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE
    )

    print("\nTraining SVM...")
    svm_model.fit(
        X_train_scaled,
        y_train
    )

    svm_results, svm_predictions, svm_cm = evaluate_model(
        svm_model,
        X_test_scaled,
        y_test,
        "SVM"
    )

    joblib.dump(
        {
            "model": svm_model,
            "scaler": scaler
        },
        os.path.join(MODEL_DIR, "face_svm.pkl")
    )

    # ---------------------------------------------
    # MODEL 2: RANDOM FOREST
    # ---------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    rf_model.fit(
        X_train,
        y_train
    )

    rf_results, rf_predictions, rf_cm = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    joblib.dump(
        rf_model,
        os.path.join(MODEL_DIR, "face_random_forest.pkl")
    )

    # ---------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------

    results_df = pd.DataFrame([
        svm_results,
        rf_results
    ])

    results_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "face_model_comparison.csv"
        ),
        index=False
    )

    pair_info["SVM_Prediction"] = svm_predictions
    pair_info["RF_Prediction"] = rf_predictions

    pair_info.to_csv(
        os.path.join(
            RESULT_DIR,
            "face_test_predictions.csv"
        ),
        index=False
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "face_svm_confusion_matrix.npy"
        ),
        svm_cm
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "face_rf_confusion_matrix.npy"
        ),
        rf_cm
    )

    print("\n========== FACE VERIFICATION COMPLETE ==========")
    print("Models saved in:", MODEL_DIR)
    print("Results saved in:", RESULT_DIR)
    print("=================================================")


if __name__ == "__main__":
    main()

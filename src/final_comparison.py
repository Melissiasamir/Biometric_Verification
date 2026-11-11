import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "comparison"
OUTPUT_FILE = OUTPUT_DIR / "final_model_comparison.csv"

COLUMNS = [
    "Modality",
    "Model",
    "Feature_Mode",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "FAR",
    "FRR",
    "TN",
    "FP",
    "FN",
    "TP",
    "Source_File",
]


def to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def to_int(value):
    if value is None or value == "":
        return None
    return int(float(value))


def read_rows(relative_path):
    path = RESULTS / relative_path

    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No results found in: {path}")

    return rows


def normalize_row(row, modality, feature_mode, source_file):
    tn = to_int(row.get("TN"))
    fp = to_int(row.get("FP"))
    fn = to_int(row.get("FN"))
    tp = to_int(row.get("TP"))

    if None in (tn, fp, fn, tp):
        raise ValueError(
            f"Missing confusion-matrix values in {source_file}, "
            f"model: {row.get('Model')}"
        )

    genuine_total = tp + fn
    impostor_total = tn + fp

    if genuine_total == 0 or impostor_total == 0:
        raise ValueError(
            f"Invalid confusion matrix in {source_file}, "
            f"model: {row.get('Model')}"
        )

    recall = tp / genuine_total
    specificity = tn / impostor_total
    far = fp / impostor_total
    frr = fn / genuine_total

    precision = to_float(row.get("Precision"))
    if precision is None:
        precision = tp / (tp + fp) if (tp + fp) else 0.0

    f1 = to_float(row.get("F1_Score"))
    if f1 is None:
        f1 = to_float(row.get("F1"))
    if f1 is None:
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

    accuracy = to_float(row.get("Accuracy"))
    if accuracy is None:
        accuracy = (tn + tp) / (tn + fp + fn + tp)

    balanced_accuracy = to_float(row.get("Balanced_Accuracy"))
    if balanced_accuracy is None:
        balanced_accuracy = (recall + specificity) / 2

    return {
        "Modality": modality,
        "Model": row.get("Model", ""),
        "Feature_Mode": row.get("Feature_Mode") or feature_mode,
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
        "TP": tp,
        "Source_File": source_file,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = []

    sources = [
        (
            "face/face_model_comparison.csv",
            "Face-only",
            "HOG + LBP",
        ),
        (
            "voice/voice_static_mfcc_rms_zcr_model_comparison.csv",
            "Voice-only",
            "Static MFCC + ZCR + RMS",
        ),
        (
            "fusion/fusion_model_comparison.csv",
            "Early Fusion",
            "Face + Voice",
        ),
        (
            "hierarchical/hierarchical_model_comparison.csv",
            "Hierarchical",
            "Face-first, Voice-fallback",
        ),
    ]

    for relative_path, modality, feature_mode in sources:
        rows = read_rows(relative_path)

        for row in rows:
            final_rows.append(
                normalize_row(
                    row,
                    modality,
                    feature_mode,
                    relative_path.replace("\\", "/"),
                )
            )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(final_rows)

    print("Final comparison created successfully.")
    print(f"Total model results: {len(final_rows)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nResults by modality:")
    for modality in dict.fromkeys(row["Modality"] for row in final_rows):
        count = sum(
            1 for row in final_rows
            if row["Modality"] == modality
        )
        print(f"{modality}: {count} models")


if __name__ == "__main__":
    main()

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "comparison" / "final_model_comparison.csv"
OUTPUT_FILE = ROOT / "results" / "comparison" / "evaluation_audit.csv"

TOLERANCE = 1e-6

METRICS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "FAR",
    "FRR",
]


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_metrics(row):
    tn = int(row["TN"])
    fp = int(row["FP"])
    fn = int(row["FN"])
    tp = int(row["TP"])

    if min(tn, fp, fn, tp) < 0:
        raise ValueError("Confusion matrix contains negative values.")

    total = tn + fp + fn + tp

    if total == 0:
        raise ValueError("Confusion matrix is empty.")

    accuracy = safe_divide(tn + tp, total)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)

    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    balanced_accuracy = (recall + specificity) / 2

    far = safe_divide(fp, fp + tn)
    frr = safe_divide(fn, fn + tp)

    return {
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score,
        "FAR": far,
        "FRR": frr,
        "Total_Pairs": total,
        "Genuine_Pairs": tp + fn,
        "Impostor_Pairs": tn + fp,
    }


def audit_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Final comparison file not found: {INPUT_FILE}\n"
            "Run src/final_comparison.py first."
        )

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The final comparison file is empty.")

    audit_rows = []

    for row in rows:
        calculated = calculate_metrics(row)
        mismatches = []

        for metric in METRICS:
            reported = float(row[metric])
            recomputed = calculated[metric]

            if abs(reported - recomputed) > TOLERANCE:
                mismatches.append(metric)

        status = "PASS" if not mismatches else "CHECK"

        audit_rows.append({
            "Modality": row["Modality"],
            "Model": row["Model"],
            "Accuracy": calculated["Accuracy"],
            "Balanced_Accuracy": calculated["Balanced_Accuracy"],
            "Precision": calculated["Precision"],
            "Recall": calculated["Recall"],
            "F1_Score": calculated["F1_Score"],
            "FAR": calculated["FAR"],
            "FRR": calculated["FRR"],
            "Total_Pairs": calculated["Total_Pairs"],
            "Genuine_Pairs": calculated["Genuine_Pairs"],
            "Impostor_Pairs": calculated["Impostor_Pairs"],
            "Audit_Status": status,
            "Mismatched_Metrics": ", ".join(mismatches),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(audit_rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    passed = sum(
        row["Audit_Status"] == "PASS"
        for row in audit_rows
    )

    failed = len(audit_rows) - passed

    print("=" * 55)
    print("BIOMETRIC VERIFICATION - EVALUATION AUDIT")
    print("=" * 55)

    print(f"Models checked: {len(audit_rows)}")
    print(f"Passed: {passed}")
    print(f"Need checking: {failed}")

    print("\nDetailed results:")

    for row in audit_rows:
        print(
            f"{row['Modality']} | "
            f"{row['Model']} | "
            f"{row['Audit_Status']}"
        )

        if row["Mismatched_Metrics"]:
            print(
                f"  Mismatched metrics: "
                f"{row['Mismatched_Metrics']}"
            )

    print("\nEvaluation audit saved to:")
    print(OUTPUT_FILE)

    if failed:
        print("\nWARNING: Some metrics do not match their confusion matrices.")
    else:
        print("\nSUCCESS: All reported metrics match their confusion matrices.")


if __name__ == "__main__":
    audit_results()

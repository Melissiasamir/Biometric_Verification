
from pathlib import Path
import csv
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "custom_dataset"

# Supported file formats
SUPPORTED_FORMATS = {
    "face": {".jpg", ".jpeg", ".png", ".bmp"},
    "voice": {".wav"},
    "fingerprint": {".bmp", ".jpg", ".jpeg", ".png"}
}


def scan_dataset():
    """Find all collected biometric samples."""

    samples = []

    for modality, extensions in SUPPORTED_FORMATS.items():
        modality_dir = DATASET_ROOT / modality

        if not modality_dir.exists():
            print(f"Warning: Missing folder: {modality_dir}")
            continue

        for person_dir in modality_dir.iterdir():

            if not person_dir.is_dir():
                continue

            person_id = person_dir.name

            for file_path in person_dir.rglob("*"):

                if file_path.is_file() and file_path.suffix.lower() in extensions:

                    samples.append({
                        "person_id": person_id,
                        "modality": modality,
                        "sample_id": str(
                            file_path.relative_to(person_dir).with_suffix("")
                        ),
                        "file_path": str(
                            file_path.relative_to(PROJECT_ROOT)
                        ),
                        "format": file_path.suffix.lower(),
                        "captured_at": datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat(timespec="seconds")
                    })

    return samples


def create_manifest(samples):
    """Save sample information in a CSV file."""

    output_file = DATASET_ROOT / "manifest.csv"

    columns = [
        "person_id",
        "modality",
        "sample_id",
        "file_path",
        "format",
        "captured_at"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nManifest saved to: {output_file}")
    return output_file


def display_dataset_summary(samples):
    """Display the number of samples collected."""

    print("\n========== CUSTOM DATASET SUMMARY ==========")

    for modality in SUPPORTED_FORMATS:
        modality_samples = [
            sample for sample in samples
            if sample["modality"] == modality
        ]

        people = set(
            sample["person_id"] for sample in modality_samples
        )

        print(f"\n{modality.upper()}")
        print(f"Total samples: {len(modality_samples)}")
        print(f"Number of participants: {len(people)}")

        for person in sorted(people):
            count = sum(
                1 for sample in modality_samples
                if sample["person_id"] == person
            )
            print(f"  {person}: {count} samples")

    if len(set(sample["person_id"] for sample in samples)) < 2:
        print(
            "\nNote: Only one participant found. "
            "Impostor rejection cannot be evaluated yet."
        )

    print("\n============================================")


def main():
    samples = scan_dataset()

    if not samples:
        print("No biometric samples found yet.")
        print("Add samples inside the appropriate participant folders.")
        return

    display_dataset_summary(samples)
    create_manifest(samples)


if __name__ == "__main__":
    main()

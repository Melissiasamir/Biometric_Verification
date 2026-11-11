from pathlib import Path
import csv
import random
from collections import defaultdict

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "vidtimit" / "manifest.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "vidtimit" / "splits.csv"


def create_splits():
    persons = defaultdict(list)

    # Read manifest
    with open(MANIFEST_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            persons[row["Person_ID"]].append(row)

    split_records = []

    # Split each person's recordings
    for person_id in sorted(persons):
        recordings = persons[person_id].copy()

        if len(recordings) != 10:
            raise ValueError(
                f"{person_id} has {len(recordings)} recordings instead of 10."
            )

        random.shuffle(recordings)

        training = recordings[:4]
        enrollment = recordings[4:6]
        testing = recordings[6:10]

        for split_name, selected_recordings in [
            ("train", training),
            ("enrollment", enrollment),
            ("test", testing),
        ]:
            for recording in selected_recordings:
                split_records.append({
                    "Person_ID": person_id,
                    "Sequence_ID": recording["Sequence_ID"],
                    "Split": split_name,
                    "Audio_Path": recording["Audio_Path"],
                    "Face_Path": recording["Face_Path"],
                    "Frame_Count": recording["Frame_Count"],
                })

    # Save split file
    fieldnames = [
        "Person_ID",
        "Sequence_ID",
        "Split",
        "Audio_Path",
        "Face_Path",
        "Frame_Count",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_records)

    # Summary
    print("\n========== DATA SPLIT SUMMARY ==========")
    print("Unique persons:", len(persons))
    print("Total recordings:", len(split_records))

    for split_name in ["train", "enrollment", "test"]:
        count = sum(1 for row in split_records if row["Split"] == split_name)
        print(f"{split_name.capitalize()} recordings:", count)

    print("Random seed:", RANDOM_SEED)
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    create_splits()
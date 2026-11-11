from pathlib import Path
import csv

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = PROJECT_ROOT / "data" / "vidtimit" / "videos"
OUTPUT_FILE = PROJECT_ROOT / "data" / "vidtimit" / "manifest.csv"


def create_manifest():
    records = []
    missing_audio = []
    missing_video = []

    # Loop through all person folders
    for person_dir in sorted(VIDEOS_DIR.iterdir()):

        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        person_root = person_dir / person_id

        audio_dir = person_root / "audio"
        video_dir = person_root / "video"

        if not audio_dir.exists() or not video_dir.exists():
            print(f"Missing audio/video folder for: {person_id}")
            continue

        # Match each WAV with its corresponding video sequence
        for audio_file in sorted(audio_dir.glob("*.wav")):

            sequence_id = audio_file.stem
            face_dir = video_dir / sequence_id

            if not face_dir.exists():
                missing_video.append((person_id, sequence_id))
                continue

            frames = [
                file for file in face_dir.iterdir()
                if file.is_file()
            ]

            if not frames:
                missing_video.append((person_id, sequence_id))
                continue

            records.append({
                "Person_ID": person_id,
                "Sequence_ID": sequence_id,
                "Audio_Path": str(audio_file.resolve()),
                "Face_Path": str(face_dir.resolve()),
                "Frame_Count": len(frames)
            })

    # Save manifest
    fieldnames = [
        "Person_ID",
        "Sequence_ID",
        "Audio_Path",
        "Face_Path",
        "Frame_Count"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Display summary
    print("\n========== MANIFEST SUMMARY ==========")
    print("Total paired sequences:", len(records))
    print("Unique persons:", len(set(r["Person_ID"] for r in records)))
    print("Missing video sequences:", len(missing_video))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    create_manifest()
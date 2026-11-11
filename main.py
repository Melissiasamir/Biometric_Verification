from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

STAGES = {
    "1": ("Data Preparation - Create Manifest", "data_preparation.py"),
    "2": ("Create Dataset Splits", "create_splits.py"),
    "3": ("Face Feature Extraction", "face_features.py"),
    "4": ("Face Verification", "face_verification.py"),
    "5": ("Voice Feature Extraction", "voice_features.py"),
    "6": ("Voice Verification", "voice_verification.py"),
    "7": ("Early Fusion Feature Creation", "fusion.py"),
    "8": ("Early Fusion Verification", "fusion_verification.py"),
    "9": ("Hierarchical Verification", "hierarchical.py"),
    "10": ("Final Model Comparison", "final_comparison.py"),
    "11": ("Evaluation Audit", "evaluation.py"),
}


def display_menu():
    print("\n" + "=" * 55)
    print("  BIOMETRIC VERIFICATION SYSTEM")
    print("=" * 55)
    print("\nSelect a project stage:\n")

    for key, (description, _) in STAGES.items():
        print(f"  {key:>2}. {description}")

    print("\n   0. Exit")
    print("=" * 55)


def run_stage(choice):
    description, filename = STAGES[choice]
    script_path = SRC_DIR / filename

    if not script_path.is_file():
        print(f"\nERROR: Script not found: {script_path}")
        return

    print(f"\nSelected stage: {description}")
    print(f"Running: {filename}")

    confirmation = input("Do you want to continue? (y/n): ").strip().lower()

    if confirmation != "y":
        print("Execution cancelled.")
        return

    print("\n" + "-" * 55)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=False
        )

        print("-" * 55)

        if result.returncode == 0:
            print(f"\nSUCCESS: {description} completed.")
        else:
            print(
                f"\nERROR: {description} failed "
                f"with exit code {result.returncode}."
            )

    except KeyboardInterrupt:
        print("\nExecution interrupted by the user.")

    except Exception as error:
        print(f"\nUnexpected error: {error}")


def main():
    print("\nWelcome to the Biometric Verification Project.")

    while True:
        display_menu()
        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("\nExiting the project. Goodbye!")
            break

        if choice not in STAGES:
            print("\nInvalid choice. Please select a valid option.")
            continue

        run_stage(choice)

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()

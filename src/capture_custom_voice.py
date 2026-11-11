
from pathlib import Path
from datetime import datetime
import time

import sounddevice as sd
import soundfile as sf
import numpy as np


# ================= CONFIGURATION =================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "custom_dataset"
    / "voice"
    / "person_02"
)

SAMPLE_RATE = 32000
DURATION = 6
MICROPHONE_ID = 1
NUMBER_OF_SAMPLES = 10

PROMPTS = [
    "My voice is being recorded for a biometric verification project.",
    "Today I am testing a speaker recognition system.",
    "I study computer science and enjoy learning new technologies.",
    "Voice recognition compares patterns in human speech.",
    "I am collecting multiple samples for my university project.",
    "Every recording should sound clear and natural.",
    "I will speak at a comfortable and consistent volume.",
    "This system uses machine learning for voice verification.",
    "I am testing the accuracy of my voice recordings.",
    "This is another sample of my natural speaking voice."
]


# ================= RECORDING FUNCTION =================

def record_sample(sample_number, session_time):

    prompt = PROMPTS[sample_number - 1]

    print(f"\n========== RECORDING {sample_number}/{NUMBER_OF_SAMPLES} ==========")
    print(f"Say: {prompt}")

    input("\nPress Enter when you are ready...")

    for countdown in [3, 2, 1]:
        print(countdown)
        time.sleep(1)

    print("\nRECORDING... Speak now!")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MICROPHONE_ID
    )

    sd.wait()

    # Calculate audio quality indicators
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    # Generate unique filename
    filename = f"voice_{session_time}_{sample_number:02d}.wav"
    output_path = OUTPUT_DIR / filename

    # Save WAV recording
    sf.write(output_path, audio, SAMPLE_RATE)

    print(f"\nSaved: {filename}")
    print(f"RMS level: {rms:.4f}")
    print(f"Peak level: {peak:.4f}")

    if rms < 0.005:
        print("WARNING: Recording may be too quiet. Check your microphone.")

    if peak > 0.98:
        print("WARNING: Recording may be distorted. Speak slightly farther away.")

    return output_path


# ================= MAIN FUNCTION =================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n========== CUSTOM VOICE COLLECTION ==========")
    print("Microphone: Built-in Realtek")
    print(f"Microphone ID: {MICROPHONE_ID}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: 1 (Mono)")
    print(f"Duration per recording: {DURATION} seconds")
    print(f"Number of recordings: {NUMBER_OF_SAMPLES}")

    print("\nINSTRUCTIONS:")
    print("1. Find a quiet environment.")
    print("2. Keep the same distance from your laptop.")
    print("3. Speak naturally and clearly.")
    print("4. Read the displayed sentence during recording.")
    print("5. Avoid background noise.")

    input("\nPress Enter to begin voice collection...")

    saved_files = []

    for i in range(1, NUMBER_OF_SAMPLES + 1):

        try:
            output_path = record_sample(i, session_time)
            saved_files.append(output_path)

        except Exception as error:
            print(f"\nRecording {i} failed: {error}")
            print("Stopping collection to avoid incomplete recordings.")
            break

    # Final summary
    print("\n========== COLLECTION SUMMARY ==========")
    print(f"Successfully recorded: {len(saved_files)}/{NUMBER_OF_SAMPLES}")
    print(f"Output folder: {OUTPUT_DIR}")

    if saved_files:
        print("\nSaved files:")
        for file_path in saved_files:
            print(f"  {file_path.name}")

    print("\n========================================")


if __name__ == "__main__":
    main()


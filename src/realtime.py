from pathlib import Path
import csv

import cv2
import librosa
import numpy as np
import sounddevice as sd

from skimage.feature import hog, local_binary_pattern


# ============================================================
# CUSTOM REAL-TIME BIOMETRIC VERIFICATION
# Uses the same custom face/voice feature pipelines and
# deterministic enrollment split as the offline verifiers.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACE_FEATURES_FILE = (
    PROJECT_ROOT / "features" / "custom_face" / "custom_face_features.npy"
)
FACE_METADATA_FILE = (
    PROJECT_ROOT / "features" / "custom_face" / "custom_face_metadata.csv"
)

VOICE_FEATURES_FILE = (
    PROJECT_ROOT / "features" / "custom_voice" / "custom_voice_features.npy"
)
VOICE_METADATA_FILE = (
    PROJECT_ROOT / "features" / "custom_voice" / "custom_voice_metadata.csv"
)

RANDOM_STATE = 42
ENROLLMENT_COUNT = 6
VALIDATION_COUNT = 2

FACE_SIZE = (128, 128)
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

SAMPLE_RATE = 32000
RECORD_SECONDS = 6
MICROPHONE_ID = 1  # Same Realtek microphone used during collection


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-10)


def cosine_similarity(a, b):
    return float(np.dot(normalize(a), normalize(b)))


def select_threshold(scores, labels):
    unique_scores = np.sort(np.unique(scores))
    candidates = [
        unique_scores[0] - 1e-9,
        unique_scores[-1] + 1e-9,
    ]
    for i in range(len(unique_scores) - 1):
        candidates.append((unique_scores[i] + unique_scores[i + 1]) / 2)

    best_threshold = None
    best_key = None
    for threshold in candidates:
        predictions = scores >= threshold
        genuine = labels == 1
        impostor = labels == 0
        far = np.mean(predictions[impostor]) if np.any(impostor) else 0
        frr = np.mean(~predictions[genuine]) if np.any(genuine) else 0
        key = (abs(far - frr), (far + frr) / 2)
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def load_metadata(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def create_split(metadata):
    """Reproduce the same per-person split used by the offline verifiers."""
    person_ids = sorted({row["Person_ID"] for row in metadata})
    if len(person_ids) != 2:
        raise ValueError(
            f"Expected exactly 2 participants; found {len(person_ids)}: {person_ids}"
        )

    rng = np.random.default_rng(RANDOM_STATE)
    enrollment, validation = {}, {}

    for person_id in person_ids:
        indices = np.array(
            [i for i, row in enumerate(metadata) if row["Person_ID"] == person_id]
        )
        if len(indices) < ENROLLMENT_COUNT + VALIDATION_COUNT + 1:
            raise ValueError(f"Not enough samples for {person_id}.")

        rng.shuffle(indices)
        enrollment[person_id] = indices[:ENROLLMENT_COUNT]
        validation[person_id] = indices[
            ENROLLMENT_COUNT:ENROLLMENT_COUNT + VALIDATION_COUNT
        ]

    return person_ids, enrollment, validation


def build_face_model():
    if not FACE_FEATURES_FILE.exists() or not FACE_METADATA_FILE.exists():
        raise FileNotFoundError(
            "Custom face features are missing. Run: python src/custom_face_features.py"
        )

    features = np.load(FACE_FEATURES_FILE)
    metadata = load_metadata(FACE_METADATA_FILE)
    if len(features) != len(metadata):
        raise ValueError("Face feature and metadata counts do not match.")

    person_ids, enrollment, validation = create_split(metadata)

    templates = {}
    for person_id in person_ids:
        samples = features[enrollment[person_id]]
        templates[person_id] = normalize(
            np.mean([normalize(row) for row in samples], axis=0)
        )

    scores, labels = [], []
    for actual_id in person_ids:
        for index in validation[actual_id]:
            for claimed_id in person_ids:
                scores.append(cosine_similarity(features[index], templates[claimed_id]))
                labels.append(int(actual_id == claimed_id))

    threshold = select_threshold(np.asarray(scores), np.asarray(labels))
    return person_ids, templates, threshold


def build_voice_model():
    if not VOICE_FEATURES_FILE.exists() or not VOICE_METADATA_FILE.exists():
        raise FileNotFoundError(
            "Custom voice features are missing. Run: python src/custom_voice_features.py"
        )

    features = np.load(VOICE_FEATURES_FILE)
    metadata = load_metadata(VOICE_METADATA_FILE)
    if len(features) != len(metadata):
        raise ValueError("Voice feature and metadata counts do not match.")

    person_ids, enrollment, validation = create_split(metadata)

    # Match custom_voice_verification.py: standardize using enrollment only.
    all_enrollment = np.concatenate(
        [features[enrollment[person_id]] for person_id in person_ids]
    )
    mean = all_enrollment.mean(axis=0)
    std = all_enrollment.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (features - mean) / std

    templates = {}
    for person_id in person_ids:
        samples = standardized[enrollment[person_id]]
        templates[person_id] = normalize(
            np.mean([normalize(row) for row in samples], axis=0)
        )

    scores, labels = [], []
    for actual_id in person_ids:
        for index in validation[actual_id]:
            for claimed_id in person_ids:
                scores.append(
                    cosine_similarity(standardized[index], templates[claimed_id])
                )
                labels.append(int(actual_id == claimed_id))

    threshold = select_threshold(np.asarray(scores), np.asarray(labels))
    return person_ids, templates, threshold, mean, std


def extract_face_features(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10),
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= lbp_hist.sum() + 1e-7

    return np.concatenate([hog_features, lbp_hist]).astype(np.float32)


def live_face_capture():
    detector = cv2.CascadeClassifier(FACE_CASCADE)
    if detector.empty():
        raise RuntimeError("Could not load the Haar Cascade face detector.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not access webcam (camera index 0).")

    print("\nFACE CHECK")
    print("Look at the camera. Press SPACE to capture when exactly one face is visible.")
    print("Press Q to cancel.")

    captured_face = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()  # Keep overlays out of the biometric crop.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            status = (
                "One face detected - SPACE to capture"
                if len(faces) == 1
                else "Keep exactly one face in view"
            )
            cv2.putText(
                frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2
            )
            cv2.imshow("Live Face Verification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                if len(faces) != 1:
                    print("Capture rejected: exactly one face must be visible.")
                    continue
                x, y, w, h = faces[0]
                captured_face = clean_frame[y:y + h, x:x + w].copy()
                captured_face = cv2.resize(captured_face, FACE_SIZE)
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if captured_face is None:
        raise RuntimeError("Face capture cancelled.")
    return extract_face_features(captured_face)


def extract_voice_features_from_array(audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("The microphone returned an empty recording.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = []
    for matrix in (mfcc, delta, delta2):
        features.extend(np.mean(matrix, axis=1))
        features.extend(np.std(matrix, axis=1))

    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=2048, hop_length=512
    )
    rms = librosa.feature.rms(
        y=audio, frame_length=2048, hop_length=512
    )
    features.extend([
        np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)
    ])
    return np.asarray(features, dtype=np.float32)


def live_voice_capture():
    print("\nVOICE CHECK")
    print(f"Speak naturally for {RECORD_SECONDS} seconds after recording starts.")
    input("Press Enter when ready...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MICROPHONE_ID,
    )
    sd.wait()
    audio = audio.reshape(-1)

    rms_level = float(np.sqrt(np.mean(audio ** 2)))
    print(f"Recording complete. Audio RMS: {rms_level:.5f}")
    if rms_level < 0.005:
        print("Warning: audio is very quiet; try again closer to the microphone.")

    return extract_voice_features_from_array(audio)


def verify(score, threshold):
    return score >= threshold


def main():
    print("\n========== REAL-TIME CUSTOM BIOMETRIC VERIFICATION ==========")
    print("This is a one-to-one verification: you must claim an identity.")
    print("Available participants are loaded from the custom dataset.")

    print("\nChoose modality:")
    print("1. Face (webcam)")
    print("2. Voice (microphone)")
    print("3. Both (face AND voice required)")
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice not in {"1", "2", "3"}:
        print("Invalid choice.")
        return

    face_model = voice_model = None
    if choice in {"1", "3"}:
        face_model = build_face_model()
    if choice in {"2", "3"}:
        voice_model = build_voice_model()

    available_ids = (
        face_model[0] if face_model is not None else voice_model[0]
    )
    if face_model is not None and voice_model is not None:
        if face_model[0] != voice_model[0]:
            raise ValueError("Face and voice participant IDs do not match.")
    print("\nRegistered identities:", ", ".join(available_ids))

    claimed_id = input("Enter the identity you are claiming (e.g. person_01): ").strip()
    if claimed_id not in available_ids:
        print("Unknown identity. Verification stopped.")
        return

    decisions = []

    if face_model is not None:
        _, templates, threshold = face_model
        live_features = live_face_capture()
        score = cosine_similarity(live_features, templates[claimed_id])
        accepted = verify(score, threshold)
        decisions.append(accepted)
        print("\nFACE RESULT")
        print(f"Claimed identity: {claimed_id}")
        print(f"Similarity score: {score:.4f}")
        print(f"Validation threshold: {threshold:.4f}")
        print("Decision:", "ACCEPTED" if accepted else "REJECTED")

    if voice_model is not None:
        _, templates, threshold, mean, std = voice_model
        live_features = live_voice_capture()
        standardized = (live_features - mean) / std
        score = cosine_similarity(standardized, templates[claimed_id])
        accepted = verify(score, threshold)
        decisions.append(accepted)
        print("\nVOICE RESULT")
        print(f"Claimed identity: {claimed_id}")
        print(f"Similarity score: {score:.4f}")
        print(f"Validation threshold: {threshold:.4f}")
        print("Decision:", "ACCEPTED" if accepted else "REJECTED")

    print("\n========== FINAL DECISION ==========")
    if all(decisions):
        print(f"VERIFIED: {claimed_id}")
    else:
        print("NOT VERIFIED")
    print("Fusion rule: AND (every selected modality must accept)")
    print("====================================")
    print("\nNote: This prototype has no face anti-spoofing or voice replay detection.")
    print("Do not use it as production-grade security or to authorize real actions.")


if __name__ == "__main__":
    main()


from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = (
    PROJECT_ROOT
    / "data"
    / "custom_dataset"
    / "face"
    / "person_02"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Face detector
CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")


class FaceCaptureApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Custom Face Data Collection")
        self.root.geometry("750x650")

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            raise RuntimeError("Could not access the webcam.")

        self.captured_count = 0
        self.target_count = 10
        self.running = True

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.preview = tk.Label(root)
        self.preview.pack(pady=10)

        self.status = tk.Label(
            root,
            text="Position your face inside the camera view.",
            font=("Arial", 12)
        )
        self.status.pack(pady=5)

        self.counter = tk.Label(
            root,
            text="Captured: 0 / 10",
            font=("Arial", 14, "bold")
        )
        self.counter.pack(pady=5)

        self.capture_button = tk.Button(
            root,
            text="Capture Face",
            command=self.capture_face,
            font=("Arial", 12),
            width=20
        )
        self.capture_button.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_camera()

    def update_camera(self):
        if not self.running:
            return

        ret, frame = self.camera.read()
        frame = cv2.flip(frame, 1)

        if ret:
            self.current_frame = frame.copy()

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    rgb_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            image = Image.fromarray(rgb_frame)
            image = image.resize((640, 480))

            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo)

            if len(faces) == 1:
                self.status.config(
                    text="Face detected. You can capture a sample."
                )
            elif len(faces) == 0:
                self.status.config(text="No face detected.")
            else:
                self.status.config(
                    text="Multiple faces detected. Keep only one person in view."
                )

        self.root.after(30, self.update_camera)

    def capture_face(self):
        if not hasattr(self, "current_frame"):
            return

        gray = cv2.cvtColor(
            self.current_frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) != 1:
            messagebox.showwarning(
                "Capture unavailable",
                "Make sure exactly one face is visible."
            )
            return

        x, y, w, h = faces[0]

        face = self.current_frame[y:y+h, x:x+w]

        face = cv2.resize(face, (128, 128))

        filename = (
            f"face_{self.session_id}_"
            f"{self.captured_count + 1:02d}.jpg"
        )

        output_path = SAVE_DIR / filename

        cv2.imwrite(str(output_path), face)

        self.captured_count += 1

        self.counter.config(
            text=f"Captured: {self.captured_count} / {self.target_count}"
        )

        if self.captured_count >= self.target_count:
            self.capture_button.config(state=tk.DISABLED)
            self.status.config(
                text="10 samples captured successfully!"
            )
            messagebox.showinfo(
                "Complete",
                f"10 face samples saved in:\n{SAVE_DIR}"
            )

    def close(self):
        self.running = False
        self.camera.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()


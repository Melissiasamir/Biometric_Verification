from pathlib import Path
from collections import Counter
import cv2

root = Path("data/fingerprint/SOCOFing/Altered")
files = list(root.rglob("*.BMP"))

print("Alteration counts:")
print(Counter(f.stem.rsplit("_", 1)[-1] for f in files))

print("\nSample image dimensions:")
for file in files[:10]:
    image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    print(file.name, image.shape if image is not None else "FAILED TO READ")

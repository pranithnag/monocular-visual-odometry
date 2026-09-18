import cv2
from pathlib import Path
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_image(sequence: str, frame: int):
    image = cv2.imread(
        str(DATA_DIR / sequence / "image_0" / f"{frame:06d}.png"))
    return image


img = load_image("00", 500)
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(img, None)

kp = keypoints[0]

print(kp.pt)
print(kp.angle)
print(kp.size)

print(descriptors.shape)
print(descriptors.dtype)
print(descriptors[0])

img_with_kp = cv2.drawKeypoints(
    img, keypoints, None, color=(0, 255, 0), flags=0)
plt.imshow(img_with_kp)
plt.title("ORB Keypoints")
plt.axis('off')
plt.show()

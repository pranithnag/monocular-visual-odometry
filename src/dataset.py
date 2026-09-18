import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_image(sequence: str, frame: int):
    image = cv2.imread(
        str(DATA_DIR / sequence / "image_0" / f"{frame:06d}.png"))
    return image


img0 = load_image("00", 0)
img1 = load_image("00", 1)

orb = cv2.ORB_create(nfeatures=500)
kps0, des0 = orb.detectAndCompute(img0, None)
kps1, des1 = orb.detectAndCompute(img1, None)

kp = kps0[0]

img0_with_kp = cv2.drawKeypoints(
    img0, kps0, None, color=(0, 255, 0), flags=0)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des0, des1, k=2)
# for i in np.arange(0.6, 1.0, 0.05):
#     good_matches = []
#     # best < threshold * second_best
#     for pair in matches:
#         if pair[0].distance < i * pair[1].distance:
#             good_matches.append(pair[0])
#     print(f'{i}: {len(good_matches)}')
good_matches = []
threshold = 0.75
# best < threshold * second_best
for pair in matches:
    if pair[0].distance < threshold * pair[1].distance:
        good_matches.append(pair[0])
print(f'{threshold}: {len(good_matches)}')
res = cv2.drawMatches(img0, kps0, img1, kps1, good_matches, None, flags=2)
cv2.imwrite(str(RESULTS_DIR / "orb_matches.png"), res)
plt.imshow(res)
plt.title("ORB Matches")
plt.axis('off')
plt.show()

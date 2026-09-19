import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_image(sequence: str, frame: int):
    image = cv2.imread(
        str(DATA_DIR / sequence / "image_0" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return image


def load_calibration(sequence: str):
    calib_file = DATA_DIR / sequence / "calib.txt"
    with open(calib_file, "r") as f:
        lines = f.readlines()
    P0 = np.array([float(x)
                  for x in lines[0].strip().split()[1:]]).reshape(3, 4)
    return P0[:, :3]


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
for match in good_matches:
    pt0 = kps0[match.queryIdx].pt
    pt1 = kps1[match.trainIdx].pt
points0 = np.array(
    [kps0[match.queryIdx].pt for match in good_matches], dtype=np.float32)
points1 = np.array(
    [kps1[match.trainIdx].pt for match in good_matches], dtype=np.float32)
print(points0.shape, points1.shape)
K = load_calibration("00")
E, mask = cv2.findEssentialMat(
    points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=1)
print(f"Essential matrix:\n{E}")
matches_mask = mask.ravel().astype(int).tolist()
res = cv2.drawMatches(img0, kps0, img1, kps1, good_matches,
                      None, matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
num_inliers, R, t, pose_mask = cv2.recoverPose(
    E, points0, points1, K, mask=mask.copy())
print("Recovered pose:")
print(f"Rotation matrix:\n{R}")
print(f"Translation vector:\n{t}")
cv2.imwrite(str(RESULTS_DIR / "ransac_inliers.png"), res)
print(f"RANSAC inliers: {np.count_nonzero(mask)}")
print(f"Pose inliers: {num_inliers}")
plt.imshow(res)
plt.title("RANSAC Inliers")
plt.axis('off')
plt.show()

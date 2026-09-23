import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_image(sequence: str, frame: int):
    image = cv2.imread(
        str(DATA_DIR / "sequences" / sequence / "image_0" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return image


def load_calibration(sequence: str):
    calib_file = DATA_DIR / "sequences" / sequence / "calib.txt"
    with open(calib_file, "r") as f:
        lines = f.readlines()
    P0 = np.array([float(x)
                  for x in lines[0].strip().split()[1:]]).reshape(3, 4)
    return P0[:, :3]


def load_ground_truth(sequence: str):
    gt_file = DATA_DIR / "poses" / f"{sequence}.txt"
    poses = []
    with open(gt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            pose = np.array([float(x)
                            for x in line.strip().split()]).reshape(3, 4)
            poses.append(pose)
    return poses


def estimate_motion(img0, img1, K):
    orb = cv2.ORB_create(nfeatures=3000)

    kps0, des0 = orb.detectAndCompute(img0, None)
    kps1, des1 = orb.detectAndCompute(img1, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des0, des1, k=2)

    good_matches = []
    threshold = 0.75

    for pair in matches:
        if len(pair) == 2 and pair[0].distance < threshold * pair[1].distance:
            good_matches.append(pair[0])

    points0 = np.array(
        [kps0[m.queryIdx].pt for m in good_matches],
        dtype=np.float32
    )

    points1 = np.array(
        [kps1[m.trainIdx].pt for m in good_matches],
        dtype=np.float32
    )

    E, mask = cv2.findEssentialMat(
        points0,
        points1,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    ransac_inliers = np.count_nonzero(mask)

    num_pose_inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        points0,
        points1,
        K,
        mask=mask.copy()
    )

    return R, t, len(good_matches), ransac_inliers, num_pose_inliers


def get_scale(gt_poses, frame):
    position0 = gt_poses[frame][:, 3]
    position1 = gt_poses[frame + 1][:, 3]
    difference = position1 - position0
    scale = np.linalg.norm(difference)
    return scale


cv2.setRNGSeed(0)
cv2.setNumThreads(1)

img0 = load_image("00", 0)
img1 = load_image("00", 1)
K = load_calibration("00")
gt_poses = load_ground_truth("00")
scale = get_scale(gt_poses, 0)
# print(f"Ground truth scale: {scale}")
# R, t, inliers = estimate_motion(img0, img1, K)
R_global = np.eye(3)
t_global = np.zeros((3, 1))
trajectory = []
for frame in range(100):
    img0 = load_image("00", frame)
    img1 = load_image("00", frame + 1)
    R, t, good_matches, essential_inliers, pose_inliers = estimate_motion(img0, img1, K)
    scale = get_scale(gt_poses, frame)
    print(f"Frame {frame}: scale={scale}")
    # if scale > 0.1:
    R_relative_inv = R.T
    t_relative_inv = -R.T @ t

    t_global = t_global + scale * (R_global @ t_relative_inv)
    R_global = R_global @ R_relative_inv
    print(f"Global translation:\n{t_global.flatten()}")
    print(f"Truth translation:\n{gt_poses[frame + 1][:, 3]}")
    print("\n")
    trajectory.append(t_global.flatten())
errors = []
for frame in range(len(trajectory)):
    estimated_position = trajectory[frame]
    truth_position = gt_poses[frame + 1][:, 3]
    error = np.linalg.norm(estimated_position - truth_position)
    errors.append(error)
# Include frame 0 and select the same frames for both plotted trajectories.
estimated_positions = np.vstack([np.zeros(3), np.asarray(trajectory)])
truth_positions = np.array([pose[:, 3] for pose in gt_poses[:len(estimated_positions)]])
position_errors = np.linalg.norm(estimated_positions - truth_positions, axis=1)
position_rmse = np.sqrt(np.mean(position_errors ** 2))

# plt.plot(errors)

# plt.xlabel("Frame")
# plt.ylabel("Position Error (m)")
# plt.title("Visual Odometry Position Error")
# plt.grid()

# plt.savefig("results/position_error.png")
# plt.show()
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(truth_positions[:, 0], truth_positions[:, 2],
        label="KITTI ground truth", color="tab:blue", linewidth=2)
ax.plot(estimated_positions[:, 0], estimated_positions[:, 2],
        label="ORB VO + GT step magnitudes", color="tab:orange", linewidth=1.5)
ax.scatter([0], [0], marker="o", color="black", label="Shared frame 0 origin")
ax.set(xlabel="X in camera 0 frame (m)", ylabel="Z in camera 0 frame (m)",
       title=f"KITTI 00 image_0 | {len(trajectory)} transitions / {len(estimated_positions)} positions\n"
             f"3D position RMSE {position_rmse:.3f} m; endpoint {position_errors[-1]:.3f} m")
ax.axis("equal")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.savefig("results/vo_truth_comparison.png")
plt.show()

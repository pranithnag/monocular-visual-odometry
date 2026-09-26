import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import time


@dataclass
class FrameResult:
    frame: int
    position: np.ndarray
    position_error: float
    good_matches: int | None = None
    essential_inliers: int | None = None
    pose_inliers: int | None = None
    inlier_ratio: float | None = None
    processing_time: float | None = None


@dataclass
class VOResult:
    method: str
    frames: list[FrameResult]

    @property
    def trajectory(self):
        return np.array([frame.position for frame in self.frames])

    @property
    def errors(self):
        return np.array([frame.position_error for frame in self.frames])

    @property
    def processing_times(self):
        return np.array(
            [
                frame.processing_time
                for frame in self.frames
                if frame.processing_time is not None
            ]
        )

    @property
    def mean_processing_time(self):
        return np.mean(self.processing_times)

    @property
    def fps(self):
        return 1 / self.mean_processing_time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_image(sequence: str, frame: int):
    image = cv2.imread(
        str(DATA_DIR / "sequences" / sequence / "image_0" / f"{frame:06d}.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    return image


def load_calibration(sequence: str):
    calib_file = DATA_DIR / "sequences" / sequence / "calib.txt"
    with open(calib_file, "r") as f:
        lines = f.readlines()
    P0 = np.array([float(x) for x in lines[0].strip().split()[1:]]).reshape(3, 4)
    return P0[:, :3]


def load_ground_truth(sequence: str):
    gt_file = DATA_DIR / "poses" / f"{sequence}.txt"
    poses = []
    with open(gt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            pose = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
            poses.append(pose)
    return poses


def estimate_motion(img0, img1, K, method="ORB", return_debug=False):
    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=3000)
    elif method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=3000)

    kps0, des0 = detector.detectAndCompute(img0, None)
    kps1, des1 = detector.detectAndCompute(img1, None)
    norm = cv2.NORM_L2
    if method == "ORB":
        norm = cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(des0, des1, k=2)

    good_matches = []
    threshold = 0.65

    for pair in matches:
        if len(pair) == 2 and pair[0].distance < threshold * pair[1].distance:
            good_matches.append(pair[0])

    points0 = np.array([kps0[m.queryIdx].pt for m in good_matches], dtype=np.float32)

    points1 = np.array([kps1[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    E, mask = cv2.findEssentialMat(
        points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=0.5
    )

    ransac_inliers = np.count_nonzero(mask)

    num_pose_inliers, R, t, pose_mask = cv2.recoverPose(
        E, points0, points1, K, mask=mask.copy()
    )
    return {
        "R": R,
        "t": t,
        "good_match_count": len(good_matches),
        "ransac_inliers": ransac_inliers,
        "pose_inliers": num_pose_inliers,
        "kps0": kps0,
        "kps1": kps1,
        "good_matches": good_matches,
        "mask": mask,
    }


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

good_match_counts = []
essential_inlier_counts = []
pose_inlier_counts = []
position_errors = []


def run_vo(method):
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    frames = [
        FrameResult(
            frame=0,
            position=np.zeros(3),
            position_error=0.0,
        )
    ]

    for frame in tqdm(range(100), desc=f"Running {method} VO"):
        img0 = load_image("00", frame)
        img1 = load_image("00", frame + 1)
        start = time.perf_counter()
        result = estimate_motion(img0, img1, K, method=method)
        elapsed = time.perf_counter() - start
        R = result["R"]
        t = result["t"]

        good_matches = result["good_match_count"]
        essential_inliers = result["ransac_inliers"]
        pose_inliers = result["pose_inliers"]

        scale = get_scale(gt_poses, frame)

        R_relative_inv = R.T
        t_relative_inv = -R.T @ t

        t_global = t_global + scale * (R_global @ t_relative_inv)

        R_global = R_global @ R_relative_inv

        position = t_global.flatten().copy()

        truth_position = gt_poses[frame + 1][:, 3]

        error = np.linalg.norm(position - truth_position)

        frames.append(
            FrameResult(
                frame=frame + 1,
                position=position,
                position_error=error,
                good_matches=good_matches,
                essential_inliers=essential_inliers,
                pose_inliers=pose_inliers,
                inlier_ratio=essential_inliers / good_matches,
                processing_time=elapsed,
            )
        )

    return VOResult(method=method, frames=frames)


orb_results = run_vo("ORB")
sift_results = run_vo("SIFT")

# Extract X and Z positions
orb_x = orb_results.trajectory[:, 0]
orb_z = orb_results.trajectory[:, 2]

sift_x = sift_results.trajectory[:, 0]
sift_z = sift_results.trajectory[:, 2]

# Ground truth positions corresponding to frames 0 -> 100
truth_positions = np.array([pose[:, 3] for pose in gt_poses[: len(orb_results.frames)]])

truth_x = truth_positions[:, 0]
truth_z = truth_positions[:, 2]


# Calculate position errors for ORB and SIFT
orb_position_errors = np.linalg.norm(orb_results.trajectory - truth_positions, axis=1)
sift_position_errors = np.linalg.norm(sift_results.trajectory - truth_positions, axis=1)

orb_error_increase = np.diff(orb_position_errors)
sift_error_increase = np.diff(sift_position_errors)

worst_frames = np.argsort(orb_error_increase)[-5:]
for frame in worst_frames:
    print(f"Worst Frame: {frame} -> {frame + 1}")
    print(f"ORB Good Matches: {orb_results.frames[frame].good_matches}")
    print(f"ORB Essential Inliers: {orb_results.frames[frame].essential_inliers}")
    print(
        f"ORB Translation Direction: {orb_results.trajectory[frame + 1] - orb_results.trajectory[frame]}"
    )
    print("--------------------------------------------------")
    print(f"SIFT Good Matches: {sift_results.frames[frame].good_matches}")
    print(f"SIFT Essential Inliers: {sift_results.frames[frame].essential_inliers}")
    print(
        f"SIFT Translation Direction: {sift_results.trajectory[frame + 1] - sift_results.trajectory[frame]}"
    )
    print("--------------------------------------------------")
worst_frame_orb = worst_frames[-1]
sift_position_errors = np.linalg.norm(sift_results.trajectory - truth_positions, axis=1)

orb_rmse = np.sqrt(np.mean(orb_position_errors**2))
sift_rmse = np.sqrt(np.mean(sift_position_errors**2))

# ============================================================
# TRAJECTORY COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    truth_x,
    truth_z,
    label="KITTI Ground Truth",
    linewidth=2,
)

ax.plot(
    orb_x,
    orb_z,
    label="ORB VO + GT Step Magnitudes",
    linewidth=1.5,
)

ax.plot(
    sift_x,
    sift_z,
    label="SIFT VO + GT Step Magnitudes",
    linewidth=1.5,
)

ax.scatter(
    [0],
    [0],
    marker="o",
    label="Frame 0 Origin",
)

ax.set(
    xlabel="X in Camera 0 Frame (m)",
    ylabel="Z in Camera 0 Frame (m)",
    title=(
        f"KITTI 00 Monocular Visual Odometry\n"
        f"ORB RMSE: {orb_rmse:.3f} m | "
        f"SIFT RMSE: {sift_rmse:.3f} m"
    ),
)

ax.axis("equal")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

plt.savefig(
    RESULTS_DIR / "orb_vs_sift_trajectory.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# POSITION ERROR COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    orb_position_errors,
    label="ORB",
)

ax.plot(
    sift_position_errors,
    label="SIFT",
)

ax.set(
    xlabel="Frame",
    ylabel="Position Error (m)",
    title="ORB vs SIFT Position Error",
)

ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

plt.savefig(
    RESULTS_DIR / "orb_vs_sift_position_error.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# RESULTS
# ============================================================

print("\nORB")
print("Final Position Error:", orb_position_errors[-1])
print("Mean Position Error:", np.mean(orb_position_errors))
print("Position RMSE:", orb_rmse)
print(
    f"ORB: {orb_results.mean_processing_time * 1000:.2f} ms/frame "
    f"({orb_results.fps:.2f} FPS)"
)
print(f"FPS: {orb_results.fps:.2f}")

print("\nSIFT")
print("Final Position Error:", sift_position_errors[-1])
print("Mean Position Error:", np.mean(sift_position_errors))
print("Position RMSE:", sift_rmse)
print(
    f"SIFT: {sift_results.mean_processing_time * 1000:.2f} ms/frame "
    f"({sift_results.fps:.2f} FPS)"
)
print(f"FPS: {sift_results.fps:.2f}")

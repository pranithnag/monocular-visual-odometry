import time
import cv2
import numpy as np
from tqdm import tqdm

from .dataset import load_image, load_calibration, load_ground_truth, get_scale
from .features import estimate_motion, is_motion_valid
from .results import FrameResult, VOResult, calculate_quality_score

cv2.setRNGSeed(0)
cv2.setNumThreads(1)
K = load_calibration("00")
gt_poses = load_ground_truth("00")

def run_vo(method, num_frames=100, reject_outliers=True):
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    frames = [
        FrameResult(
            frame=0,
            position=np.zeros(3),
            position_error=0.0,
        )
    ]
    reference_frame = 0
    reference_image = load_image("00", 0)
    reference_R, reference_t = R_global.copy(), t_global.copy()
    pending_scale = 0.0
    for frame in tqdm(range(num_frames), desc=f"Running {method} VO"):
        img0 = reference_image
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

        pending_scale += scale
        valid = is_motion_valid(result)
        used_reference = reference_frame
        if valid:
            # eference -> current, not previous -> current.
            t_global = reference_t + pending_scale * (reference_R @ (-R.T @ t))
            R_global = reference_R @ R.T
            reference_frame, reference_image = frame + 1, img1
            reference_R, reference_t = R_global.copy(), t_global.copy()
            pending_scale = 0.0
        elif result["rotation_valid"]:
            R_global = reference_R @ R.T
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
                inlier_ratio=essential_inliers / max(good_matches, 1),
                pose_inlier_ratio=pose_inliers / max(good_matches, 1),
                processing_time=elapsed,
                quality_score=calculate_quality_score(
                    good_matches, essential_inliers, pose_inliers
                ),
                accepted=valid,
                rotation_updated=result["rotation_valid"],
                reference_frame=used_reference,
            )
        )

    return VOResult(method=method, frames=frames)

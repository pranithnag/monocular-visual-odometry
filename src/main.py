import matplotlib.pyplot as plt
import numpy as np

from .dataset import RESULTS_DIR, load_image, get_scale
from .features import estimate_motion
from .results import log_results
from .vo import K, gt_poses, run_vo


if __name__ == "__main__":
    img0 = load_image("00", 0)
    img1 = load_image("00", 1)
    scale = get_scale(gt_poses, 0)
    # print(f"Ground truth scale: {scale}")
    # R, t, inliers = estimate_motion(img0, img1, K)

    good_match_counts = []
    essential_inlier_counts = []
    pose_inlier_counts = []
    position_errors = []

    sift_results = run_vo("SIFT", num_frames=1000)
    frame = 540

    img0 = load_image("00", frame)
    img1 = load_image("00", frame + 1)

    result = estimate_motion(img0, img1, K, method="SIFT")

    print("Good matches:", result["good_match_count"])
    print("RANSAC inliers:", result["ransac_inliers"])
    print("Pose inliers:", result["pose_inliers"])

    print(
        "RANSAC ratio:",
        result["ransac_inliers"] / result["good_match_count"]
    )

    print(
        "Pose ratio:",
        result["pose_inliers"] / result["good_match_count"]
    )

    print("R:")
    print(result["R"])

    print("t:")
    print(result["t"].flatten())

    log_results(sift_results, "SIFT")
    sift_x = sift_results.trajectory[:, 0]
    sift_z = sift_results.trajectory[:, 2]

    # Ground truth positions corresponding to frames 0 -> 100
    truth_positions = np.array(
        [pose[:, 3] for pose in gt_poses[: len(sift_results.frames)]]
    )

    truth_x = truth_positions[:, 0]
    truth_z = truth_positions[:, 2]


    # Calculate position errors for SIFT
    sift_position_errors = np.linalg.norm(sift_results.trajectory - truth_positions, axis=1)

    sift_error_increase = np.diff(sift_position_errors)

    worst_frames = np.argsort(sift_error_increase)[-5:]
    # for frame in worst_frames:
    # print(f"Worst Frame: {frame} -> {frame + 1}")
    # # print(f"ORB Good Matches: {orb_results.frames[frame].good_matches}")
    # # print(f"ORB Essential Inliers: {orb_results.frames[frame].essential_inliers}")
    # # print(
    # #     f"ORB Translation Direction: {orb_results.trajectory[frame + 1] - orb_results.trajectory[frame]}"
    # # )
    # print("--------------------------------------------------")
    # print(f"SIFT Good Matches: {sift_results.frames[frame].good_matches}")
    # print(f"SIFT Essential Inliers: {sift_results.frames[frame].essential_inliers}")
    # print(
    #     f"SIFT Translation Direction: {sift_results.trajectory[frame + 1] - sift_results.trajectory[frame]}"
    # )
    # print(f"SIFT Position Error Increase: {sift_error_increase[frame]}")
    # print(f"SIFT Quality Score: {sift_results.frames[frame].quality_score}")
    # print("--------------------------------------------------\n")
    sift_position_errors = np.linalg.norm(sift_results.trajectory - truth_positions, axis=1)

    sift_rmse = np.sqrt(np.mean(sift_position_errors**2))
    ratios = np.array([frame.pose_inlier_ratio for frame in sift_results.frames[1:]])


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
        title=(f"KITTI 00 Monocular Visual Odometry\n" f"SIFT RMSE: {sift_rmse:.3f} m"),
    )

    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    plt.savefig(
        RESULTS_DIR / "final_trajectory.png",
        dpi=300,
    )

    plt.close(fig)


    # ============================================================
    # POSITION ERROR COMPARISON
    # ============================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        sift_position_errors,
        label="SIFT",
    )

    ax.set(
        xlabel="Frame",
        ylabel="Position Error (m)",
        title="SIFT Position Error",
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    plt.savefig(
        RESULTS_DIR / "final_position_error.png",
        dpi=300,
    )

    plt.close(fig)


    # ============================================================
    # RESULTS
    # ============================================================
    # rejected = [frame.frame for frame in orb_results.frames if frame.accepted is False]
    # print("Rejected frames:", rejected)
    # print("Number rejected:", len(rejected))
    # print("\nORB")
    # print("Final Position Error:", orb_position_errors[-1])
    # print("Mean Position Error:", np.mean(orb_position_errors))
    # print("Position RMSE:", orb_rmse)
    # print(
    #     f"ORB: {orb_results.mean_processing_time * 1000:.2f} ms/frame "
    #     f"({orb_results.fps:.2f} FPS)"
    # )
    # print(f"FPS: {orb_results.fps:.2f}")

    print(f"Frames: {len(sift_results.frames) - 1}")
    print(f"RMSE: {sift_results.rmse:.3f} m")
    print(f"Mean error: {sift_results.mean_error:.3f} m")
    print(f"Final error: {sift_results.final_error:.3f} m")
    print(f"Mean processing time: {sift_results.mean_processing_time * 1000:.2f} ms")
    print(f"FPS: {sift_results.fps:.2f}")

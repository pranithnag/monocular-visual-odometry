"""VO result records, metrics, and console reporting."""
from dataclasses import dataclass
import numpy as np

@dataclass
class FrameResult:
    frame: int
    position: np.ndarray
    position_error: float
    good_matches: int | None = None
    essential_inliers: int | None = None
    pose_inliers: int | None = None
    inlier_ratio: float | None = None
    pose_inlier_ratio: float | None = None
    processing_time: float | None = None
    quality_score: float | None = None
    accepted: bool | None = None
    rotation_updated: bool | None = None
    reference_frame: int | None = None


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

    @property
    def rmse(self):
        return np.sqrt(np.mean(self.errors**2))

    @property
    def mean_error(self):
        return np.mean(self.errors)

    @property
    def final_error(self):
        return self.errors[-1]


def calculate_quality_score(
    good_matches,
    ransac_inliers,
    pose_inliers,
):
    if good_matches == 0:
        return 0.0

    match_score = min(good_matches / 1000.0, 1.0)

    essential_ratio = ransac_inliers / good_matches
    pose_ratio = pose_inliers / good_matches

    quality_score = 0.2 * match_score + 0.4 * essential_ratio + 0.4 * pose_ratio

    return quality_score


def log_results(vo_results, method_name):
    print(f"\n{method_name} Results:")
    print("Final Position Error:", vo_results.errors[-1])
    print("Mean Position Error:", np.mean(vo_results.errors))
    print("Position RMSE:", np.sqrt(np.mean(vo_results.errors**2)))
    print(
        f"{method_name}: {vo_results.mean_processing_time * 1000:.2f} ms/frame "
        f"({vo_results.fps:.2f} FPS)"
    )
    print(
        f"Mean Inlier Ratio: {np.mean([frame.inlier_ratio for frame in vo_results.frames[1:]])}"
    )
    print(
        f"Mean Good Matches: {np.mean([frame.good_matches for frame in vo_results.frames[1:] if frame.good_matches is not None])}"
    )
    print(f"FPS: {vo_results.fps:.2f}")

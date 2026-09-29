from pathlib import Path
import cv2
import numpy as np

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


def get_scale(gt_poses, frame):
    position0 = gt_poses[frame][:, 3]
    position1 = gt_poses[frame + 1][:, 3]
    difference = position1 - position0
    scale = np.linalg.norm(difference)
    return scale

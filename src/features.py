import cv2
import numpy as np


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
    matches = bf.knnMatch(des0, des1, k=2) if des0 is not None and des1 is not None and len(des1) >= 2 else []

    good_matches = []
    threshold = 0.65

    for pair in matches:
        if len(pair) == 2 and pair[0].distance < threshold * pair[1].distance:
            good_matches.append(pair[0])

    points0 = np.array([kps0[m.queryIdx].pt for m in good_matches], dtype=np.float32)

    points1 = np.array([kps1[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    E, mask = (cv2.findEssentialMat(
        points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=0.5
    ) if len(good_matches) >= 5 else (None, None))

    ransac_inliers = np.count_nonzero(mask)

    motion = recover_motion(E, points0, points1, K, mask)
    return {
        **motion,
        "good_match_count": len(good_matches),
        "ransac_inliers": ransac_inliers,
        "kps0": kps0,
        "kps1": kps1,
        "good_matches": good_matches,
        "mask": mask,
    }


def recover_motion(E, points0, points1, K, mask):
    empty = dict(R=np.eye(3), t=np.zeros((3, 1)), pose_inliers=0,
                 translation_valid=False, rotation_valid=False)
    if E is None or mask is None or len(points0) < 5:
        return empty
    E = np.asarray(E)
    if E.ndim != 2 or E.shape[1] != 3 or E.shape[0] % 3 or not np.isfinite(E).all():
        return empty
    mask = np.asarray(mask, dtype=np.uint8).reshape(-1, 1).copy()
    selected = mask.ravel() != 0
    if selected.sum() < 5:
        return empty
    rays = [] 
    for points in (points0, points1):
        xy = cv2.undistortPoints(points.astype(np.float64)[:, None], K, None).reshape(-1, 2)
        ray = np.column_stack((xy, np.ones(len(xy))))
        rays.append(ray / np.linalg.norm(ray, axis=1, keepdims=True))
    a, b = rays
    best = None
    for candidate in E.reshape(-1, 3, 3):
        singular = np.linalg.svd(candidate, compute_uv=False)
        if singular[0] == 0 or singular[2] > 1e-3 * singular[0] or abs(singular[0]-singular[1]) > 1e-3 * singular[0]:
            continue
        count, R, t, pose_mask, _ = cv2.recoverPose(
            candidate, points0, points1, K, distanceThresh=1e9, mask=mask.copy())
        if best is None or count > best[0]:
            best = count, R, t, pose_mask
    if best is None:
        return empty
    count, R, t, pose_mask = best
    parallax = np.arccos(np.clip(np.sum(a * (b @ R), axis=1), -1, 1))
    # Three times the angular equivalent of the existing 0.5-pixel RANSAC
    # tolerance. Positive depth alone cannot establish a usable baseline.
    noise = np.arctan(0.5 / min(K[0, 0], K[1, 1]))
    supported = (pose_mask.ravel() != 0) & (parallax >= 3 * noise)
    translation_valid = (supported.sum() >= 20 and supported.sum() >= 0.1 * selected.sum()
                         and count > 0 and np.median(parallax[pose_mask.ravel() != 0]) >= 3 * noise)
    result = dict(R=R, t=t, pose_inliers=int(count),
                  translation_valid=bool(translation_valid), rotation_valid=bool(translation_valid))
    if translation_valid:
        return result
    weights = selected.astype(float)
    for _ in range(10):
        U, _, Vt = np.linalg.svd((b * weights[:, None]).T @ a)
        rotation = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        predicted = (a @ rotation.T) @ K.T
        with np.errstate(divide='ignore', invalid='ignore'):
            residual = np.linalg.norm(predicted[:, :2] / predicted[:, 2:] - points1, axis=1)
        residual[(predicted[:, 2] <= 0) | ~np.isfinite(residual)] = np.inf
        weights = selected * np.minimum(1., 0.5 / np.maximum(residual, 1e-9))
    rotation_support = selected & (residual <= 0.5)
    if rotation_support.sum() >= 20 and rotation_support.sum() >= 0.5 * selected.sum():
        result.update(R=rotation, rotation_valid=True)
    return result


def is_motion_valid(result):
    return result["translation_valid"]

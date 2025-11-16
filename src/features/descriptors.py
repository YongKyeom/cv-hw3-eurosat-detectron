from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from utils.io import to_gray

# ---------------------------------------------------------------------------
# 🔵 1. Patch Raw Descriptor (기본 픽셀 평탄화 + z-score 정규화)
# ---------------------------------------------------------------------------


def desc_patch_raw(patch: np.ndarray) -> np.ndarray:
    """
    패치를 벡터로 평탄화한 뒤 z-score 정규화하여 반환한다.

    Args:
        patch (np.ndarray): 입력 패치(BGR 또는 Gray), shape=(H, W[, 3]).

    Returns:
        np.ndarray: (D,) 형태의 float32 벡터.

    Notes:
        - 조명/대비 차이를 줄이기 위해 (x - mean) / std 수행.
        - 기하 변화(회전/스케일)에는 약함.
    """
    vec = patch.astype(np.float32).reshape(-1)
    mean = float(vec.mean())
    std = float(vec.std() + 1e-6)  # 0 division 방지
    return ((vec - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# 🔵 2. HOG (Histogram of Oriented Gradients) Descriptor
# ---------------------------------------------------------------------------


def desc_hog(
    patch: np.ndarray,
    num_cells: int = 2,
    bins: int = 8,
) -> np.ndarray:
    """
    HOG 디스크립터를 계산한다.

    Args:
        patch (np.ndarray): 패치(BGR 또는 Gray).
        num_cells (int): 가로/세로 셀 개수. (2면 2×2 셀)
        bins (int): 방향 히스토그램 bin 개수(0~180°).

    Returns:
        np.ndarray: (num_cells*num_cells*bins,) float32 벡터.

    Notes:
        - Gray 변환 후 Sobel로 gradient 계산 → magnitude / orientation.
        - 각 셀 별로 magnitude 가중치 히스토그램 생성.
        - 조명 변화에 비교적 강건.
    """
    g = to_gray(patch)

    # Sobel 미분
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ori = cv2.phase(gx, gy, angleInDegrees=True) % 180.0

    h, w = g.shape
    cell_h, cell_w = h // num_cells, w // num_cells
    desc: List[float] = []

    for cy in range(num_cells):
        for cx in range(num_cells):
            ys, ye = cy * cell_h, (cy + 1) * cell_h if cy < num_cells - 1 else h
            xs, xe = cx * cell_w, (cx + 1) * cell_w if cx < num_cells - 1 else w

            cell_mag = mag[ys:ye, xs:xe].reshape(-1)
            cell_ori = ori[ys:ye, xs:xe].reshape(-1)

            hist = np.zeros((bins,), dtype=np.float32)
            bin_width = 180.0 / bins
            idx = np.clip((cell_ori / bin_width).astype(np.int32), 0, bins - 1)

            # magnitude 가중치로 누적
            for i in range(len(idx)):
                hist[idx[i]] += cell_mag[i]

            # L2 정규화
            hist /= float(np.linalg.norm(hist)) + 1e-6
            desc.extend(hist.tolist())

    return np.asarray(desc, dtype=np.float32)


# ---------------------------------------------------------------------------
# 🔵 3. HSV Color Histogram Descriptor
# ---------------------------------------------------------------------------


def desc_color_hist_hsv(
    patch: np.ndarray,
    h_bins: int = 16,
    s_bins: int = 8,
    v_bins: int = 8,
) -> np.ndarray:
    """
    HSV 3D 컬러 히스토그램 디스크립터를 계산한다.

    Args:
        patch (np.ndarray): 입력 패치(BGR).
        h_bins (int): H 채널 bin 개수(0~180).
        s_bins (int): S 채널 bin 개수(0~256).
        v_bins (int): V 채널 bin 개수(0~256).

    Returns:
        np.ndarray: 1D float32 벡터(H×S×V).

    Notes:
        - 조명 변화(밝기)에 비교적 강건(H/S 중심).
        - 배경/무채색 패치에서는 성능이 떨어질 수 있음.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.astype(np.float32).reshape(-1)
    hist /= float(hist.sum()) + 1e-6  # L1 정규화
    return hist


# ---------------------------------------------------------------------------
# 🔵 4. SIFT Descriptor (KeyPoint 기반)
# ---------------------------------------------------------------------------


def compute_sift_descriptors(
    image: np.ndarray,
    keypoints: Sequence[cv2.KeyPoint],
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    SIFT 디스크립터를 계산한다.

    Args:
        image (np.ndarray): 입력 이미지(BGR 또는 Gray).
        keypoints (Sequence[cv2.KeyPoint]): 키포인트 리스트.

    Returns:
        Tuple[List[cv2.KeyPoint], np.ndarray]:
            - 사용된 실제 KeyPoint 리스트(필터링 후).
            - 디스크립터 행렬(N, 128).

    Notes:
        - Bag-of-Features, Feature Matching, Homography 등에 활용.
    """
    gray = to_gray(image)
    sift = cv2.SIFT_create()
    kps, desc = sift.compute(gray, list(keypoints))
    desc = desc.astype(np.float32) if desc is not None else np.zeros((0, 128), dtype=np.float32)
    return kps, desc


# ---------------------------------------------------------------------------
# 🔵 5. 여러 패치에 대해 디스크립터 일괄 계산
# ---------------------------------------------------------------------------


def compute_descriptors(
    patches: Sequence[np.ndarray],
    kind: str,
) -> np.ndarray:
    """
    패치 목록을 받아 지정된 종류의 디스크립터를 일괄 계산한다.

    Args:
        patches (Sequence[np.ndarray]): 패치 이미지 리스트.
        kind (str): {"patch", "grad", "color"}

    Returns:
        np.ndarray: (N, D) float32 디스크립터 행렬.

    Raises:
        ValueError: 지원되지 않는 kind 지정 시.
    """
    descs: List[np.ndarray] = []

    if kind == "patch":
        for p in patches:
            descs.append(desc_patch_raw(p))

    elif kind == "grad":
        for p in patches:
            descs.append(desc_hog(p))

    elif kind == "color":
        for p in patches:
            descs.append(desc_color_hist_hsv(p))

    else:
        raise ValueError(f"Unsupported descriptor type: {kind}")

    return np.vstack(descs).astype(np.float32)


# ---------------------------------------------------------------------------
# 🔵 6. 모든 디스크립터를 한 번에 반환하는 멀티-팩토리 (선택적)
# ---------------------------------------------------------------------------


def compute_all_descriptors(
    patches: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    """
    모든 디스크립터(patch/grad/color)를 한 번에 계산하여 반환한다.

    Args:
        patches (Sequence[np.ndarray]): 패치 리스트.

    Returns:
        dict[str, np.ndarray]: key="patch"/"grad"/"color" → (N,D) 행렬.

    Notes:
        - Bag-of-Features 실험에서 descriptor 변형 간 비교할 때 유용함.
    """
    return {
        "patch": compute_descriptors(patches, "patch"),
        "grad": compute_descriptors(patches, "grad"),
        "color": compute_descriptors(patches, "color"),
    }

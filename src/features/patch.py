from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def extract_grid_patches(
    image: np.ndarray,
    patch_w: int,
    patch_h: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[cv2.KeyPoint]]:
    """
    그리드 기반 슬라이딩 윈도우 방식으로 패치를 추출한다.

    Args:
        image (np.ndarray): 입력 BGR 또는 Grayscale 이미지.
        patch_w (int): 패치 너비.
        patch_h (int): 패치 높이.
        stride (int): 슬라이딩 간격(픽셀).

    Returns:
        Tuple[List[np.ndarray], List[cv2.KeyPoint]]:
            - patches: 패치(BGR 혹은 Gray) 이미지 리스트.
            - keypoints: 각 패치의 중심점 좌표를 나타내는 cv2.KeyPoint 리스트.

    Raises:
        ValueError: 패치 크기가 원본 이미지보다 큰 경우.

    Note:
        - HW2의 FeatureDescriptors.make_patches 구조를 계승.
        - Bag-of-Features, patch descriptor 기반 분석, EuroSAT MLP 입력 등 다양한 실습에 사용됨.
    """
    H, W = image.shape[:2]

    if patch_w > W or patch_h > H:
        raise ValueError(f"패치 크기({patch_w}x{patch_h})가 이미지 크기({W}x{H})보다 큽니다.")

    patches: List[np.ndarray] = []
    keypoints: List[cv2.KeyPoint] = []

    # 슬라이딩 윈도우 기반 패치 추출
    for y0 in range(0, H - patch_h + 1, stride):
        for x0 in range(0, W - patch_w + 1, stride):
            patch = image[y0 : y0 + patch_h, x0 : x0 + patch_w].copy()

            # 패치 중심 좌표 계산
            cx = x0 + patch_w / 2.0
            cy = y0 + patch_h / 2.0

            # KeyPoint는 위치(pt)와 크기(size)만 사용
            kp = cv2.KeyPoint(
                float(cx),
                float(cy),
                float((patch_w + patch_h) / 2.0),
            )

            patches.append(patch)
            keypoints.append(kp)

    return patches, keypoints


def patches_to_montage(
    patches: Sequence[np.ndarray],
    cols: int = 16,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    패치 리스트를 타일 형태의 몽타주 이미지로 변환한다.

    Args:
        patches (Sequence[np.ndarray]): 패치 이미지 배열.
        cols (int): 한 줄에 배치할 패치 수.
        gap (int): 패치 간 간격.
        bg_color (Tuple[int,int,int]): 배경색(BGR).

    Returns:
        np.ndarray: 몽타주 이미지(BGR).
    """
    if len(patches) == 0:
        # 빈 입력 보호
        return np.zeros((32, 32, 3), dtype=np.uint8)

    ph, pw = patches[0].shape[:2]
    N = len(patches)
    rows = (N + cols - 1) // cols

    H = rows * ph + (rows - 1) * gap
    W = cols * pw + (cols - 1) * gap

    canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

    for idx, patch in enumerate(patches):
        r = idx // cols
        c = idx % cols
        y = r * (ph + gap)
        x = c * (pw + gap)
        canvas[y : y + ph, x : x + pw] = patch

    return canvas


def save_patches(
    patches: Sequence[np.ndarray],
    keypoints: Sequence[cv2.KeyPoint],
    out_dir: Path,
    base_name: str = "patch",
) -> None:
    """
    패치를 개별 이미지 파일로 저장한다.

    Args:
        patches (Sequence[np.ndarray]): 패치 이미지 리스트.
        keypoints (Sequence[cv2.KeyPoint]): 각 패치 중심 좌표.
        out_dir (Path): 저장할 디렉토리.
        base_name (str): 파일명 접두어.

    Note:
        - 파일명에 (y, x) 좌표를 함께 포함해 재현성을 높인다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for patch, kp in zip(patches, keypoints):
        h, w = patch.shape[:2]
        cx, cy = kp.pt
        x0 = int(round(cx - w / 2))
        y0 = int(round(cy - h / 2))
        fname = f"{base_name}_{y0}_{x0}.png"
        cv2.imwrite(str(out_dir / fname), patch)

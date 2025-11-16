from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import cv2
import numpy as np

from utils.io import save_image

# ---------------------------------------------------------------------------
# 🔵 1. Brute-Force Matching (for descriptors)
# ---------------------------------------------------------------------------


def bf_match_and_draw(
    img1: np.ndarray,
    kps1: Sequence[cv2.KeyPoint],
    desc1: np.ndarray,
    img2: np.ndarray,
    kps2: Sequence[cv2.KeyPoint],
    desc2: np.ndarray,
    save_path: Union[str, "Path"],
    norm_type: int = cv2.NORM_L2,
    do_ratio_test: bool = False,
    ratio: float = 0.75,
    topk: int | None = 200,
) -> Tuple[List[cv2.DMatch], np.ndarray]:
    """
    Brute-Force 매칭을 수행하고 시각화 이미지를 저장한다.

    Args:
        img1 (np.ndarray): 좌측 이미지 (BGR)
        kps1 (Sequence[cv2.KeyPoint]): 좌측 키포인트
        desc1 (np.ndarray): 좌측 디스크립터 행렬 (N1, D)
        img2 (np.ndarray): 우측 이미지 (BGR)
        kps2 (Sequence[cv2.KeyPoint]): 우측 키포인트
        desc2 (np.ndarray): 우측 디스크립터 행렬 (N2, D)
        save_path (str | Path): 시각화 결과 저장 파일 경로
        norm_type (int): 거리 측정 방식(cv2.NORM_L2 / cv2.NORM_HAMMING)
        do_ratio_test (bool): Lowe ratio test 수행 여부
        ratio (float): ratio test 임계값
        topk (int | None): 상위 topk개의 매칭만 시각화 (None이면 전체)

    Returns:
        Tuple[List[cv2.DMatch], np.ndarray]:
            - matches: 최종 매칭 리스트
            - vis: drawMatches 결과 이미지(BGR)
    """
    bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)

    # KNN 기반 ratio test
    if do_ratio_test:
        knn = bf.knnMatch(desc1, desc2, k=2)
        good: List[cv2.DMatch] = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        matches = sorted(good, key=lambda x: x.distance)
    else:
        # 단순 match
        matches = sorted(bf.match(desc1, desc2), key=lambda x: x.distance)

    # topk 제한
    if topk is not None:
        matches = matches[:topk]

    # drawMatches
    vis = cv2.drawMatches(
        img1,
        list(kps1),
        img2,
        list(kps2),
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    save_image(save_path, vis)

    return matches, vis


# ---------------------------------------------------------------------------
# 🔵 2. SIFT: detect → compute → ratio match
# ---------------------------------------------------------------------------


def sift_detect_and_match(
    img1: np.ndarray,
    img2: np.ndarray,
    save_path: str | "Path",
    ratio: float = 0.75,
    topk: int = 500,
) -> Tuple[
    List[cv2.KeyPoint],
    np.ndarray,
    List[cv2.KeyPoint],
    np.ndarray,
    List[cv2.DMatch],
    np.ndarray,
]:
    """
    SIFT 키포인트 추출 → 디스크립터 계산 → Lowe ratio KNN 매칭 → 매칭 시각화를 수행한다.

    Args:
        img1 (np.ndarray): 좌측 이미지(BGR)
        img2 (np.ndarray): 우측 이미지(BGR)
        save_path (str | Path): drawMatches 저장 경로
        ratio (float): Lowe ratio test 임계값
        topk (int): 상위 몇 개의 매칭만 시각화할지

    Returns:
        Tuple[
            kps1, desc1, kps2, desc2, matches, vis
        ]
    """
    sift = cv2.SIFT_create()

    # detect + compute
    kps1, desc1 = sift.detectAndCompute(img1, None)
    kps2, desc2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good: List[cv2.DMatch] = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)[:topk]

    vis = cv2.drawMatches(
        img1,
        kps1,
        img2,
        kps2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_image(save_path, vis)

    return kps1, desc1, kps2, desc2, good, vis


# ---------------------------------------------------------------------------
# 🔵 3. 매칭 결과로부터 좌표 쌍 추출 (Homography 입력)
# ---------------------------------------------------------------------------


def pts_from_matches(
    kps1: Sequence[cv2.KeyPoint],
    kps2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    매칭 리스트에서 좌표쌍(src_pts, dst_pts)을 (N,1,2) 형태로 추출한다.

    Args:
        kps1 (Sequence[cv2.KeyPoint]): 좌측 키포인트들
        kps2 (Sequence[cv2.KeyPoint]): 우측 키포인트들
        matches (Sequence[cv2.DMatch]): 매칭 리스트

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - src_pts: shape (N,1,2), float32
            - dst_pts: shape (N,1,2), float32
    """
    src = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src, dst


# ---------------------------------------------------------------------------
# 🔵 4. 매칭 거리 요약 통계
# ---------------------------------------------------------------------------


def match_distance_stats(matches: Iterable[cv2.DMatch]) -> Dict[str, float]:
    """
    매칭 거리 분포에 대한 요약 통계(n, mean, median, min, max, p90)를 계산한다.

    Args:
        matches (Iterable[cv2.DMatch]): 매칭 리스트

    Returns:
        Dict[str, float]: 요약 통계
    """
    matches = list(matches)

    if len(matches) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
        }

    dist = np.asarray([m.distance for m in matches], dtype=np.float64)

    return {
        "n": int(dist.size),
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "min": float(dist.min()),
        "max": float(dist.max()),
        "p90": float(np.quantile(dist, 0.90)),
    }


# ---------------------------------------------------------------------------
# 🔵 5. CSV 저장을 위한 행(row) 생성
# ---------------------------------------------------------------------------


def matches_to_rows(matches: Sequence[cv2.DMatch]) -> List[List[object]]:
    """
    매칭 리스트를 CSV 저장용 행(row) 리스트로 변환한다.

    Args:
        matches (Sequence[cv2.DMatch]): 매칭 리스트

    Returns:
        List[List[object]]: [rank, distance, queryIdx, trainIdx] 리스트
    """
    rows: List[List[object]] = []
    for i, m in enumerate(matches, start=1):
        rows.append([i, float(m.distance), int(m.queryIdx), int(m.trainIdx)])
    return rows

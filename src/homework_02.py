# type: ignore
"""
HW2 과제
    - 문제 1: 그리드 패치 추출, 그래디언트/컬러 히스토그램 디스크립터, BF 매칭, SIFT 매칭
    - 문제 2: SIFT 매칭 포인트로 Homography (SVD, RANSAC) 추정 후 스티칭
    - 문제 3: RANSAC 직접 구현(정규화 DLT) 및 스티칭, OpenCV RANSAC과 비교

입출력
- 입력: ./data/paired_images/*.png
    - 예시: annapurna_left_01.png / annapurna_right_01.png
- 출력: ./result/hw2/*.png, ./result/hw2/*.csv, ./result/hw2/*.txt
- 로그: ./log/hw2_*.log
"""

import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

current_path = os.path.dirname(__file__)
sys.path.append(current_path)

from utils.logger import get_logger


@dataclass
class Paths:
    """
    프로젝트 경로 모음.

    Attributes:
        log_dir (Path): 로그 디렉토리 경로
        result_dir (Path): 결과 이미지 저장 디렉토리
        pair_dir (Path): 페어 이미지 디렉토리(./data/paired_images)
    """

    log_dir: Path
    result_dir: Path
    pair_dir: Path

    @staticmethod
    def from_root(root: str | Path) -> "Paths":
        """
        루트 경로로부터 표준 경로를 생성

        Args:
            root (str | Path): 프로젝트 루트 경로

        Returns:
            Paths: 표준 경로 집합
        """
        root = Path(root).resolve()
        log = root / "log"
        result = root / "result" / "hw2"
        pair = root / "data" / "paired_images"

        log.mkdir(parents=True, exist_ok=True)
        result.mkdir(parents=True, exist_ok=True)

        return Paths(log_dir=log, result_dir=result, pair_dir=pair)


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    BGR -> Gray 변환 (이미 Gray면 그대로 반환)

    Args:
        img (np.ndarray): 입력 이미지.

    Returns:
        np.ndarray: Grayscale 이미지.

    """
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_color(path: Path | str) -> np.ndarray:
    """
    컬러 이미지를 로드

    Args:
        path (Path | str): 이미지 경로

    Returns:
        np.ndarray: BGR 컬러 이미지

    Raises:
        FileNotFoundError: 파일을 찾을 수 없을 때
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")

    return img


def save_image(path: Path | str, img: np.ndarray) -> None:
    """
    이미지 저장 (경로 자동 생성)

    Args:
        path (Path | str): 저장 경로
        img (np.ndarray): 저장할 이미지

    Returns:
        None: 반환값 없음
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

    return None


def save_csv(path: Path | str, header: List[str], rows: List[List[object]]) -> None:
    """
    CSV 파일 저장

    Args:
        path (Path | str): 저장 경로
        header (List[str]): 헤더 리스트
        rows (List[List[object]]): 데이터 행 목록

    Returns:
        None: 반환값 없음
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)

    return None


def save_stats_txt(path: Path | str, lines: List[str]) -> None:
    """
    텍스트 파일로 간단한 통계 요약을 저장

    Args:
        path (Path | str): 저장 경로
        lines (List[str]): 라인 텍스트 목록

    Returns:
        None: 반환값 없음
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return None


class FeatureDescriptors:
    """
    패치 생성과 패치 기반 Descriptors를 생성하는 모듈

    - 패치 벡터화 디스크립터(정규화)
    - HOG(셀 히스토그램)
    - HSV 컬러 히스토그램
    """

    @staticmethod
    def make_patches(image: np.ndarray, patch_w: int, patch_h: int, stride: int) -> Tuple[List[np.ndarray], List[cv2.KeyPoint]]:
        """
        그리드 기반으로 작은 패치와 해당 중심점 KeyPoint들을 생성한다.
        슬라이딩 윈도우로 (patch_h, patch_w) 크기의 패치를 stride 간격으로 자른다.
        각 패치의 원본 좌표계 기준 중심을 cv2.KeyPoint로 저장한다.

        Args:
            image (np.ndarray): 입력 BGR 이미지 (H, W, 3)
            patch_w (int): 패치 너비
            patch_h (int): 패치 높이
            stride (int): 슬라이딩 간격(픽셀)

        Returns:
            Tuple[List[np.ndarray], List[cv2.KeyPoint]]: 패치 리스트, 키포인트 리스트

        Raises:
            ValueError: 패치 크기가 이미지보다 클 때
        """
        # 1) 입력 이미지 크기 확인
        H, W = image.shape[:2]
        if patch_w > W or patch_h > H:
            raise ValueError("패치 크기가 원본보다 큽니다.")

        patches: List[np.ndarray] = []
        keypoints: List[cv2.KeyPoint] = []

        # 2) 슬라이딩 윈도우로 패치 추출
        for y0 in range(0, H - patch_h + 1, stride):
            for x0 in range(0, W - patch_w + 1, stride):
                # 2-1) 패치 슬라이스
                patch = image[y0 : y0 + patch_h, x0 : x0 + patch_w].copy()
                # 2-2) 패치 중심 좌표 계산
                cx = x0 + patch_w / 2.0
                cy = y0 + patch_h / 2.0
                # 2-3) 키포인트 객체 생성(size는 패치 대략 크기)
                kp = cv2.KeyPoint(float(cx), float(cy), size=float((patch_w + patch_h) / 2.0))
                # 2-4) 수집
                patches.append(patch)
                keypoints.append(kp)

        return patches, keypoints

    @staticmethod
    def desc_patch_raw(patch: np.ndarray) -> np.ndarray:
        """
        이미지 패치를 벡터화한 디스크립터를 생성한다.
        밝기 표준화를 위해 (x - mean) / std 정규화를 적용한 뒤 평탄화한다.

        Args:
            patch (np.ndarray): 패치 이미지 (h, w, 3)

        Returns:
            np.ndarray: 1차원 벡터(float32)
        """
        # 1) float32 변환 및 평탄화
        vec = patch.astype(np.float32).reshape(-1)
        # 2) 정규화(밝기/대비 보정)
        m, s = float(vec.mean()), float(vec.std() + 1e-6)
        vec = (vec - m) / s

        return vec

    @staticmethod
    def desc_hog(patch: np.ndarray, num_cells: int = 2, bins: int = 8) -> np.ndarray:
        """
        HOG 디스크립터를 계산한다.
        Gray 변환 후 Sobel 미분으로 magnitude/angle을 구하고,
        패치를 (num_cells x num_cells) 셀로 나눠 각 셀의 방향 히스토그램(가중: magnitude)을 만든다.

        Args:
            patch (np.ndarray): 패치 이미지 (h, w, 3 또는 1)
            num_cells (int): 가로/세로 셀 개수
            bins (int): 방향 히스토그램 bin 개수(0~180도)

        Returns:
            np.ndarray: 디스크립터 벡터(float32), 길이=(num_cells^2)*bins
        """
        # 1) Gray 변환
        g = to_gray(patch)

        # 2) Sobel로 x,y 미분 → magnitude, orientation
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ori = cv2.phase(gx, gy, angleInDegrees=True) % 180.0

        # 3) 셀 분할 크기 산정
        h, w = g.shape
        cell_h, cell_w = h // num_cells, w // num_cells

        # 4) 각 셀에서 방향 히스토그램 누적
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
                for k in range(len(idx)):
                    hist[idx[k]] += cell_mag[k]

                # 5) L2 정규화
                norm = float(np.linalg.norm(hist) + 1e-6)
                hist = hist / norm
                desc.extend(hist.tolist())

        return np.asarray(desc, dtype=np.float32)

    @staticmethod
    def desc_color_hist_hsv(patch: np.ndarray, h_bins: int = 16, s_bins: int = 8, v_bins: int = 8) -> np.ndarray:
        """
        HSV 컬러 히스토그램 디스크립터를 계산한다.

        Args:
            patch (np.ndarray): 패치 이미지 (h, w, 3)
            h_bins (int): H 채널 bin 개수(0~180)
            s_bins (int): S 채널 bin 개수(0~256)
            v_bins (int): V 채널 bin 개수(0~256)

        Returns:
            np.ndarray: 정규화된 1D 히스토그램(float32)
        """
        # 1) HSV 변환
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # 2) 3채널 히스토그램 계산
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
        hist = hist.astype(np.float32).reshape(-1)

        # 3) 정규화(합=1)
        hist_sum = float(hist.sum() + 1e-6)
        hist = hist / hist_sum

        return hist

    @staticmethod
    def compute_descriptors(patches: Sequence[np.ndarray], kind: str) -> np.ndarray:
        """
        패치 리스트에서 지정된 종류의 디스크립터 행렬을 만든다.

        Args:
            patches (Sequence[np.ndarray]): 패치 이미지 리스트
            kind (str): 디스크립터 종류 {"patch", "grad", "color"}

        Returns:
            np.ndarray: (N, D) float32 행렬

        Raises:
            ValueError: 지원하지 않는 kind일 때
        """
        descs: List[np.ndarray] = []
        # kind에 맞는 디스크립터 생성기를 적용
        if kind == "patch":
            for p in patches:
                descs.append(FeatureDescriptors.desc_patch_raw(p))
        elif kind == "grad":
            for p in patches:
                descs.append(FeatureDescriptors.desc_hog(p))
        elif kind == "color":
            for p in patches:
                descs.append(FeatureDescriptors.desc_color_hist_hsv(p))
        else:
            raise ValueError("unknown descriptor kind")

        return np.vstack(descs).astype(np.float32)

    @staticmethod
    def save_patches_as_images(
        patches: Sequence[np.ndarray],
        keypoints: Sequence[cv2.KeyPoint],
        out_dir: Path,
        base_name: str = "patch",
    ) -> None:
        """
        패치 이미지를 개별 파일로 저장한다.

        Args:
            patches (Sequence[np.ndarray]): 패치 리스트 (각 BGR 이미지)
            keypoints (Sequence[cv2.KeyPoint]): 각 패치 중심 좌표(저장 파일명 표기용)
            out_dir (Path): 저장 폴더 경로
            base_name (str): 파일명 접두어

        Returns:
            None
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p, kp in zip(patches, keypoints):
            h, w = p.shape[:2]
            cx, cy = kp.pt
            x0 = int(round(cx - w / 2.0))
            y0 = int(round(cy - h / 2.0))
            fname = f"{base_name}_{y0}_{x0}.png"
            cv2.imwrite(str(out_dir / fname), p)
        return None

    @staticmethod
    def render_patches_montage(
        patches: Sequence[np.ndarray],
        cols: int = 24,
        gap: int = 2,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        패치들을 타일 형태의 몽타주 이미지로 합친다.

        Args:
            patches (Sequence[np.ndarray]): 패치 리스트(BGR)
            cols (int): 한 줄에 배치할 열 개수
            gap (int): 타일 간격(픽셀)
            bg_color (Tuple[int,int,int]): 배경색(BGR)

        Returns:
            np.ndarray: 몽타주 BGR 이미지
        """
        if not patches:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        ph, pw = patches[0].shape[:2]
        n = len(patches)
        rows = (n + cols - 1) // cols
        H = rows * ph + (rows - 1) * gap
        W = cols * pw + (cols - 1) * gap
        canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)
        for idx, p in enumerate(patches):
            r = idx // cols
            c = idx % cols
            y = r * (ph + gap)
            x = c * (pw + gap)
            canvas[y : y + ph, x : x + pw] = p
        return canvas


class FeatureMatching:
    """
    패치, SIFT Descriptors Matching 및 시각화 유틸
    """

    @staticmethod
    def bfmatch_and_draw(
        img1: np.ndarray,
        kps1: Sequence[cv2.KeyPoint],
        desc1: np.ndarray,
        img2: np.ndarray,
        kps2: Sequence[cv2.KeyPoint],
        desc2: np.ndarray,
        out_path: Path,
        norm_type: int = cv2.NORM_L2,
        do_ratio: bool = False,
        ratio: float = 0.75,
        topk: int | None = 200,
    ) -> Tuple[List[cv2.DMatch], np.ndarray]:
        """
        Brute-Force 매칭을 수행하고 시각화 이미지를 저장한다.

        Args:
            img1 (np.ndarray): 좌측 이미지
            kps1 (Sequence[cv2.KeyPoint]): 좌측 키포인트들
            desc1 (np.ndarray): 좌측 디스크립터 (N1, D)
            img2 (np.ndarray): 우측 이미지
            kps2 (Sequence[cv2.KeyPoint]): 우측 키포인트들
            desc2 (np.ndarray): 우측 디스크립터 (N2, D)
            out_path (Path): 결과 이미지 저장 경로
            norm_type (int): 거리 척도(cv2.NORM_L2 / cv2.NORM_HAMMING)
            do_ratio (bool): Lowe ratio test 사용 여부
            ratio (float): ratio 임계값
            topk (int | None): 상위 매칭 수 제한(None이면 전체)

        Returns:
            Tuple[List[cv2.DMatch], np.ndarray]: 매칭 리스트, drawMatches 결과 이미지
        """
        # 1) BFMatcher 생성(교차검증 비활성: 더 많은 후보 확보)
        bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)

        # 2) 매칭 수행: ratio test 사용 여부 분기
        if do_ratio:
            knn = bf.knnMatch(desc1, desc2, k=2)
            good: List[cv2.DMatch] = []
            for m, n in knn:
                if m.distance < ratio * n.distance:
                    good.append(m)
            matches = sorted(good, key=lambda m: m.distance)

        else:
            m = bf.match(desc1, desc2)
            matches = sorted(m, key=lambda x: x.distance)

        # 3) 상위 topk로 제한(시각화 가독성)
        if topk is not None:
            matches = matches[:topk]

        # 4) 매칭 시각화 이미지 생성/저장
        vis = cv2.drawMatches(
            img1,
            list(kps1),
            img2,
            list(kps2),
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        save_image(out_path, vis)

        return matches, vis

    @staticmethod
    def sift_detect_and_match(
        img1: np.ndarray,
        img2: np.ndarray,
        out_path: Path,
        ratio: float = 0.75,
        topk: int = 500,
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray, List[cv2.KeyPoint], np.ndarray, List[cv2.DMatch], np.ndarray]:
        """
        SIFT로 키포인트/디스크립터를 추출하고 Lowe ratio 매칭을 수행한다.

        Args:
            img1 (np.ndarray): 좌측 이미지
            img2 (np.ndarray): 우측 이미지
            out_path (Path): drawMatches 저장 경로
            ratio (float): Lowe ratio 임계값
            topk (int): 상위 매칭 수 제한

        Returns:
            Tuple[...]: (kps1, desc1, kps2, desc2, matches, vis이미지)
        """
        # 1) SIFT 추출기 준비
        sift = cv2.SIFT_create()

        # 2) 키포인트/디스크립터 계산
        k1, d1 = sift.detectAndCompute(img1, None)
        k2, d2 = sift.detectAndCompute(img2, None)

        # 3) BF + KNN 매칭 후 Lowe ratio로 필터링
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(d1, d2, k=2)

        good: List[cv2.DMatch] = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        good = sorted(good, key=lambda m: m.distance)[:topk]

        # 4) 시각화
        vis = cv2.drawMatches(
            img1,
            k1,
            img2,
            k2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        save_image(out_path, vis)

        return k1, d1, k2, d2, good, vis

    @staticmethod
    def pts_from_matches(
        k1: Sequence[cv2.KeyPoint],
        k2: Sequence[cv2.KeyPoint],
        matches: Sequence[cv2.DMatch],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        매칭으로부터 좌표 쌍을 (N,1,2) 형태로 만든다.

        Args:
            k1 (Sequence[cv2.KeyPoint]): 좌측 키포인트
            k2 (Sequence[cv2.KeyPoint]): 우측 키포인트
            matches (Sequence[cv2.DMatch]): 매칭 리스트

        Returns:
            Tuple[np.ndarray, np.ndarray]: src_pts, dst_pts (각각 (N,1,2), float32)
        """
        # 1) queryIdx/trainIdx로 좌표 인덱스 추출
        src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        return src, dst

    @staticmethod
    def match_distance_stats(matches: Sequence[cv2.DMatch]) -> Dict[str, float]:
        """
        매칭 거리 분포 요약 통계를 계산한다.

        Args:
            matches (Sequence[cv2.DMatch]): 매칭 리스트

        Returns:
            Dict[str, float]: n, mean, median, min, max, p90 요약
        """
        if not matches:
            return {
                "n": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "p90": float("nan"),
            }
        d = np.array([m.distance for m in matches], dtype=np.float64)

        return {
            "n": int(d.size),
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "min": float(d.min()),
            "max": float(d.max()),
            "p90": float(np.quantile(d, 0.90)),
        }

    @staticmethod
    def matches_to_rows(matches: Sequence[cv2.DMatch]) -> List[List[object]]:
        """
        매칭 리스트를 CSV 저장을 위한 행 리스트로 변환한다.

        Args:
            matches (Sequence[cv2.DMatch]): 매칭 리스트

        Returns:
            List[List[object]]: [rank, distance, queryIdx, trainIdx] 행 목록
        """
        rows: List[List[object]] = []
        for i, m in enumerate(matches, start=1):
            rows.append([i, float(m.distance), int(m.queryIdx), int(m.trainIdx)])

        return rows


class HomographyToolkit:
    """
    호모그래피 추정(SVD/RANSAC), DLT, RANSAC 직접 구현, Stiching 모듈
    """

    # --- OpenCV 기반 추정 ---
    @staticmethod
    def find_homography_svd(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SVD(일반 최소자승) 방식으로 호모그래피를 구한다.

        Args:
            src_pts (np.ndarray): 원본 점들 (N,1,2)
            dst_pts (np.ndarray): 대응 점들 (N,1,2)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (H, mask)
        """
        H, mask = cv2.findHomography(src_pts, dst_pts, method=0)

        return H, mask

    @staticmethod
    def find_homography_ransac(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransacReprojThreshold: float = 3.0,
        confidence: float = 0.995,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RANSAC으로 호모그래피를 구한다.

        Args:
            src_pts (np.ndarray): 원본 점들 (N,1,2)
            dst_pts (np.ndarray): 대응 점들 (N,1,2)
            ransacReprojThreshold (float): 인라이어 판정 임계값(픽셀)
            confidence (float): 신뢰도

        Returns:
            Tuple[np.ndarray, np.ndarray]: (H, inlier_mask)
        """
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransacReprojThreshold,
            confidence=confidence,
        )

        return H, mask

    # --- 정규화 DLT/오차 ---
    @staticmethod
    def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pixel 정규화(평행이동+스케일)

        Args:
            pts (np.ndarray): (N,2) 또는 (N,1,2) 점 배열

        Returns:
            Tuple[np.ndarray, np.ndarray]: (정규화된 점 (N,2), 정규화 행렬 T(3x3))
        """
        pts = pts.reshape(-1, 2).astype(np.float64)
        c = pts.mean(axis=0)
        d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
        # Degenerate 보호: 평균 거리 d가 너무 작으면 스케일=1(평행이동만) 처리
        if not np.isfinite(d) or d < 1e-8:
            s = 1.0
            T = np.array([[1.0, 0.0, -c[0]], [0.0, 1.0, -c[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            s = np.sqrt(2.0) / d
            T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones]).T
        ptsn = T @ pts_h
        ptsn = (ptsn[:2, :] / (ptsn[2:3, :] + 1e-12)).T

        return ptsn, T

    @staticmethod
    def is_valid_homography(H: np.ndarray, cond_thresh: float = 1e12) -> bool:
        """
        호모그래피 유효성 검사

        Args:
            H (np.ndarray): 3x3 행렬
            cond_thresh (float): 조건수 임계값(클수록 허용 폭 커짐)

        Returns:
            bool: 유효하면 True
        """
        if H is None:
            return False
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            return False
        if not np.all(np.isfinite(H)):
            return False
        # 행렬 조건수 과대면 수치 불안정 가능성 큼
        try:
            c = np.linalg.cond(H)
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(c) or c > cond_thresh:
            return False
        # 투영 분모 성분이 완전히 0에 가까우면 위험
        if abs(H[2, 2]) < 1e-12:
            return False

        return True

    @staticmethod
    def _is_degenerate_sample(pts: np.ndarray, eps: float = 1e-6) -> bool:
        """
        4점 샘플이 거의 공선/면적 0에 가까운지 검사

        Args:
            pts (np.ndarray): (4,2) 또는 (4,1,2)
            eps (float): 면적 임계값

        Returns:
            bool: 퇴화 샘플이면 True
        """
        p = pts.reshape(-1, 2).astype(np.float64)
        if p.shape[0] < 4:
            return True

        def tri_area(a, b, c) -> Any:
            return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        a1 = tri_area(p[0], p[1], p[2])
        a2 = tri_area(p[1], p[2], p[3])
        a3 = tri_area(p[0], p[2], p[3])

        # 세 개 면적이 모두 매우 작으면 거의 공선
        return (a1 < eps and a2 < eps) or (a1 < eps and a3 < eps) or (a2 < eps and a3 < eps)

    @staticmethod
    def dlt_homography(src_pts: np.ndarray, dst_pts: np.ndarray, do_normalize: bool = True) -> np.ndarray:
        """
        DLT(Direct Linear Transform)로 호모그래피를 계산한다.

        Args:
            src_pts (np.ndarray): 원본 점들 (N,2) 또는 (N,1,2)
            dst_pts (np.ndarray): 대응 점들 (N,2) 또는 (N,1,2)
            do_normalize (bool): Hartley 정규화 사용 여부

        Returns:
            np.ndarray: 3x3 호모그래피 행렬(float64)
        """
        src = src_pts.reshape(-1, 2).astype(np.float64)
        dst = dst_pts.reshape(-1, 2).astype(np.float64)
        if do_normalize:
            src, Ts = HomographyToolkit.normalize_points(src)
            dst, Td = HomographyToolkit.normalize_points(dst)
        else:
            Ts = np.eye(3, dtype=np.float64)
            Td = np.eye(3, dtype=np.float64)

        A = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A = np.asarray(A, dtype=np.float64)

        # SVD 해 얻기
        try:
            _, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError as e:
            raise e
        Hn = Vt[-1, :].reshape(3, 3)
        H = np.linalg.inv(Td) @ Hn @ Ts
        # 스케일 정규화: H[2,2]가 너무 작으면 L2 노름으로 대체
        denom = H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
        H = H / (denom + 1e-12)
        # 수치 유효성 체크
        if not np.all(np.isfinite(H)):
            raise np.linalg.LinAlgError("Non-finite H from DLT")

        return H.astype(np.float64)

    @staticmethod
    def reprojection_errors(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        호모그래피 재투영 오차(L2 픽셀 거리)를 계산한다.

        Args:
            H (np.ndarray): 호모그래피 행렬(3x3)
            src_pts (np.ndarray): 원본 점들 (N,2) 또는 (N,1,2)
            dst_pts (np.ndarray): 대응 점들 (N,2) 또는 (N,1,2)

        Returns:
            np.ndarray: 각 점의 재투영 오차 (N,)
        """
        H = np.asarray(H, dtype=np.float64)
        src = src_pts.reshape(-1, 2).astype(np.float64)
        dst = dst_pts.reshape(-1, 2).astype(np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return np.full((src.shape[0],), np.inf, dtype=np.float64)
        # 과도한 스케일 방지용 정규화
        H = H / max(1e-12, np.linalg.norm(H))
        ones = np.ones((src.shape[0], 1), dtype=np.float64)
        src_h = np.hstack([src, ones])
        # (N,3) @ (3,3)^T 형태로 곱하면 같은 결과이며 수치적으로 안정적
        proj_h = src_h @ H.T
        w = proj_h[:, 2:3]
        valid = np.abs(w) > 1e-8
        proj = np.empty_like(dst)
        # 유효한 분모에 대해서만 나누기
        proj[valid[:, 0]] = proj_h[valid[:, 0], :2] / w[valid[:, 0]]
        # 무효한 경우는 큰 오차로 처리
        proj[~valid[:, 0]] = np.nan
        err = np.linalg.norm(proj - dst, axis=1)
        err[~valid[:, 0]] = np.inf

        return err

    @staticmethod
    def ransac_homography_custom(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        max_iters: int = 3000,
        thresh: float = 3.0,
        confidence: float = 0.995,
        do_normalize: bool = True,
        seed: int | None = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        직접 구현한 RANSAC으로 호모그래피를 추정한다.

        - 표본 크기: 4 (DLT 최소 표본)
        - 에러: 단방향 재투영 오차(L2, 픽셀)
        - 마지막에 인라이어로 재추정

        Args:
            src_pts (np.ndarray): 원본 점들 (N,2) 또는 (N,1,2)
            dst_pts (np.ndarray): 대응 점들 (N,2) 또는 (N,1,2)
            max_iters (int): 최대 반복 횟수
            thresh (float): 인라이어 판정 임계값(픽셀)
            confidence (float): 신뢰도(조기 종료 근거)
            do_normalize (bool): Hartley 정규화 사용 여부
            seed (int | None): 난수 시드

        Returns:
            Tuple[np.ndarray, np.ndarray]: 최종 H, 인라이어 마스크(N,1) uint8

        Raises:
            ValueError: 점 개수가 4 미만일 때
            RuntimeError: 유효한 모델을 찾지 못했을 때
        """
        rng = np.random.default_rng(seed)
        src = src_pts.reshape(-1, 2).astype(np.float64)
        dst = dst_pts.reshape(-1, 2).astype(np.float64)
        N = src.shape[0]
        if N < 4:
            raise ValueError("RANSAC에 최소 4점 필요")

        best_inliers: np.ndarray = np.zeros((N,), dtype=bool)
        best_H: np.ndarray | None = None
        best_count = 0

        # 핵심 반복: 4점 샘플링 → DLT → 모든 점 오차 → 인라이어 갱신
        for it in range(max_iters):
            idx = rng.choice(N, size=4, replace=False)
            # 퇴화 샘플 방지: src/dst 모두 검사
            if HomographyToolkit._is_degenerate_sample(src[idx]) or HomographyToolkit._is_degenerate_sample(dst[idx]):
                continue

            try:
                Hc = HomographyToolkit.dlt_homography(src[idx], dst[idx], do_normalize=do_normalize)
            except np.linalg.LinAlgError:
                continue

            # 수치 유효성 검사
            if not HomographyToolkit.is_valid_homography(Hc):
                continue

            err = HomographyToolkit.reprojection_errors(Hc, src, dst)
            inliers = err < thresh
            count = int(inliers.sum())

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_H = Hc

                # 반복 횟수 갱신
                w = max(count / N, 1e-6)
                s = 4
                denom = max(1e-12, math.log(1 - w**s))
                est_iters = math.log(1 - confidence) / denom
                if est_iters < it + 1:
                    break

        if best_H is None:
            raise RuntimeError("RANSAC 실패: 유효한 H를 찾지 못함")

        # 인라이어만으로 H 재추정
        H_final = HomographyToolkit.dlt_homography(src[best_inliers], dst[best_inliers], do_normalize=do_normalize)
        mask = best_inliers.astype(np.uint8).reshape(-1, 1)

        return H_final.astype(np.float64), mask

    @staticmethod
    def stitch_two(img1: np.ndarray, img2: np.ndarray, H_2to1: np.ndarray) -> np.ndarray:
        """
        img2를 img1 좌표계로 워핑한 뒤 간단히 블렌딩하여 파노라마를 만든다.

        - 캔버스 크기를 두 이미지의 모서리 변환으로 계산
        - 음수 좌표 보정을 위한 translation 행렬 적용
        - 겹치는 영역은 단순 평균 블렌딩

        Args:
            img1 (np.ndarray): 기준 이미지
            img2 (np.ndarray): 워핑될 이미지
            H_2to1 (np.ndarray): img2 → img1 호모그래피(3x3)

        Returns:
            np.ndarray: 스티칭 결과 이미지(BGR)
        """
        # 1) 크기/코너 정의
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 2) img2 코너를 img1 공간으로 투시변환
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H_2to1).reshape(-1, 2)

        # 3) 전체 캔버스 경계 계산
        corners = np.vstack(
            [
                warped_corners2,
                np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32),
            ]
        )
        x_min, y_min = np.floor(corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(corners.max(axis=0)).astype(int)

        # 4) 음수 좌표 보정용 이동 행렬
        tx, ty = (-x_min if x_min < 0 else 0), (-y_min if y_min < 0 else 0)
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

        # 5) 워핑 및 캔버스 합성
        out_w, out_h = int(x_max - x_min), int(y_max - y_min)
        warped2 = cv2.warpPerspective(img2, T @ H_2to1, (out_w, out_h))

        canvas = np.zeros_like(warped2)
        canvas[ty : ty + h1, tx : tx + w1] = img1

        # 6) 단순 평균 블렌딩(겹치지 않는 영역은 바로 복사)
        mask2 = (warped2 > 0).astype(np.uint8)
        mask1 = (canvas > 0).astype(np.uint8)
        overlap = (mask1 & mask2).astype(np.uint8)

        out = canvas.copy()
        out[mask2.astype(bool) & ~overlap.astype(bool)] = warped2[mask2.astype(bool) & ~overlap.astype(bool)]
        if overlap.any():
            idx = overlap.astype(bool)
            out[idx] = ((canvas[idx].astype(np.float32) + warped2[idx].astype(np.float32)) / 2.0).astype(np.uint8)

        return out

    @staticmethod
    def homography_stats(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, thresh: float = 3.0) -> Dict[str, float]:
        """
        호모그래피의 재투영오차 기반 요약 통계를 계산한다.

        Args:
            H (np.ndarray): 호모그래피(3x3)
            src_pts (np.ndarray): 소스 점들 (N,1,2)
            dst_pts (np.ndarray): 타깃 점들 (N,1,2)
            thresh (float): 인라이어 임계값(픽셀)

        Returns:
            Dict[str, float]: n, inliers, inlier_ratio, mean_err, median_err, p90_err
        """
        if H is None or not HomographyToolkit.is_valid_homography(H):
            return {
                "n": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "mean_err": float("nan"),
                "median_err": float("nan"),
                "p90_err": float("nan"),
            }
        err = HomographyToolkit.reprojection_errors(H, src_pts, dst_pts)
        n = err.size
        inl = (err < thresh).sum()

        return {
            "n": int(n),
            "inliers": int(inl),
            "inlier_ratio": float(inl / max(1, n)),
            "mean_err": float(err.mean()),
            "median_err": float(np.median(err)),
            "p90_err": float(np.quantile(err, 0.90)),
        }


def discover_pairs(pair_dir: Path, min_pairs: int = 2) -> List[Tuple[Path, Path]]:
    """
    페어 이미지 디렉토리에서 (left, right) 쌍을 자동으로 찾는다.

    파일명 패턴: {name}_left_{id}.png / {name}_right_{id}.png

    Args:
        pair_dir (Path): 페어 이미지 디렉토리
        min_pairs (int): 경고 임계치(최소 권장 페어 수)

    Returns:
        List[Tuple[Path, Path]]: (left_path, right_path) 리스트(정렬됨)
    """
    files = [p for p in pair_dir.glob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    pair_pattern = re.compile(r"^(?P<name>.+?)_(left|right)_(?P<id>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

    buckets: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for p in files:
        m = pair_pattern.match(p.name)
        if not m:
            continue
        key = (m.group("name"), m.group("id"))
        side = "left" if "left" in p.name.lower() else "right"
        buckets.setdefault(key, {})[side] = p

    pairs: List[Tuple[Path, Path]] = []
    for _, v in buckets.items():
        if "left" in v and "right" in v:
            pairs.append((v["left"], v["right"]))
    pairs = sorted(pairs)

    return pairs


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # 0) 경로/로거 설정
    # ----------------------------------------------------------------------------------
    project_root = Path(os.path.join(os.path.dirname(__file__), "..")).resolve()
    paths = Paths.from_root(project_root)
    logger = get_logger(log_dir=paths.log_dir, name="hw02")
    logger.info("===== HW2 실행 시작 =====")

    # ----------------------------------------------------------------------------------
    # 1) 페어 이미지 검색
    # ----------------------------------------------------------------------------------
    pairs = discover_pairs(paths.pair_dir, min_pairs=2)
    if not pairs:
        logger.error("paired_images 폴더에서 (left/right) 페어를 찾지 못했습니다. 파일명 패턴을 확인하세요.")
        sys.exit(1)
    logger.info("발견한 페어 수: %d", len(pairs))

    # ----------------------------------------------------------------------------------
    # 2) 문제 1 — 패치/디스크립터/매칭 결과 + 통계 산출
    # ----------------------------------------------------------------------------------
    # 2-1) 예시 패치 그리드 시각화(annapurna_left_01가 있을 때)
    sample_path = paths.pair_dir / "annapurna_left_01.png"
    if sample_path.exists():
        try:
            img = load_color(sample_path)
            patches, kps = FeatureDescriptors.make_patches(img, patch_w=32, patch_h=32, stride=24)

            # [1-1] 패치 그리드(사각형) 오버레이 저장 (원본 이미지 위 확인용)
            vis = img.copy()
            ph, pw = patches[0].shape[:2]
            for kp in kps:
                cx, cy = kp.pt
                x0 = int(round(cx - pw / 2.0))
                y0 = int(round(cy - ph / 2.0))
                x1 = x0 + pw
                y1 = y0 + ph
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
            save_image(paths.result_dir / "p1_1_patch_grid_annapurna.png", vis)
            logger.info("[1-1] 패치 그리드(사각형) 시각화 저장: p1_1_patch_grid_annapurna.png")

            # 문제 1-1: 패치 파일 저장 + 읽기 검증 몽타주
            patch_dir = paths.result_dir / "patches" / sample_path.stem
            FeatureDescriptors.save_patches_as_images(patches, kps, patch_dir, base_name="patch")

            montage = FeatureDescriptors.render_patches_montage(patches, cols=24, gap=2)
            save_image(paths.result_dir / "p1_1_patch_montage_annapurna.png", montage)

            logger.info("[1-1] 패치 개별 저장 완료")

            # ----------------------------------------------------------------------------------
            # [1-2] patch_192_24.png 기준: Gradient / Color Histogram 결과 생성 및 저장
            #  - 저장 위치: result/hw2/p1_2_gradmag_patch_192_24.png
            #             result/hw2/p1_2_patch_192_24_overview.png
            # ----------------------------------------------------------------------------------
            try:
                target_patch_path = patch_dir / "patch_192_24.png"
                if not target_patch_path.exists():
                    # 지정 패치가 없으면 첫 번째 패치로 대체
                    cand = sorted(patch_dir.glob("*.png"))
                    if cand:
                        target_patch_path = cand[0]
                        logger.warning("[1-2] 지정 패치가 없어 첫 패치를 사용: %s", target_patch_path.name)
                    else:
                        raise FileNotFoundError("패치 파일을 찾지 못했습니다.")

                # (a) 패치 로드 (BGR)
                patch_ex = cv2.imread(str(target_patch_path), cv2.IMREAD_COLOR)
                if patch_ex is None:
                    raise FileNotFoundError(f"패치 로드 실패: {target_patch_path}")

                # (b) Gradient magnitude (Sobel)
                gray = to_gray(patch_ex)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(gx, gy)
                mag_u8 = cv2.normalize(mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                save_image(paths.result_dir / "p1_2_gradmag_patch_192_24.png", mag_u8)

                # (c) Color histogram (B, G, R 각각 24 bins)
                bins = 24
                hist_b = cv2.calcHist([patch_ex], [0], None, [bins], [0, 256]).reshape(-1)
                hist_g = cv2.calcHist([patch_ex], [1], None, [bins], [0, 256]).reshape(-1)
                hist_r = cv2.calcHist([patch_ex], [2], None, [bins], [0, 256]).reshape(-1)

                # (d) Figure 구성: (1) 원본 패치 (2) Gradient magnitude (3) 컬러 히스토그램
                fig = plt.figure(figsize=(10, 3.2))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title("Original Patch")
                ax1.imshow(cv2.cvtColor(patch_ex, cv2.COLOR_BGR2RGB))
                ax1.axis("off")

                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title("Gradient Magnitude")
                ax2.imshow(mag_u8, cmap="gray")
                ax2.axis("off")

                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title("Color Histogram (B,G,R)")
                x = np.arange(bins)
                ax3.plot(x, hist_b, label="B")
                ax3.plot(x, hist_g, label="G")
                ax3.plot(x, hist_r, label="R")
                ax3.set_xlim(0, bins - 1)
                ax3.legend(loc="upper right")
                fig.tight_layout()
                fig.savefig(paths.result_dir / "p1_2_patch_192_24_overview.png", dpi=200)
                plt.close(fig)

                # (e) 디스크립터 벡터 저장: HOG(2x2x8=32D), HSV 3D 히스토그램(16x8x8=1024D)
                hog_desc = FeatureDescriptors.desc_hog(patch_ex, num_cells=2, bins=8)
                color_desc = FeatureDescriptors.desc_color_hist_hsv(patch_ex, h_bins=16, s_bins=8, v_bins=8)

                # NPY 저장
                np.save(paths.result_dir / "p1_2_desc_hog_patch_192_24.npy", hog_desc)
                np.save(paths.result_dir / "p1_2_desc_color_hsv_patch_192_24.npy", color_desc)

                # CSV 저장 (idx, value)
                save_csv(
                    paths.result_dir / "p1_2_desc_hog_patch_192_24.csv",
                    header=["idx", "value"],
                    rows=[[i, float(v)] for i, v in enumerate(hog_desc.reshape(-1))],
                )
                save_csv(
                    paths.result_dir / "p1_2_desc_color_hsv_patch_192_24.csv",
                    header=["idx", "value"],
                    rows=[[i, float(v)] for i, v in enumerate(color_desc.reshape(-1))],
                )

                logger.info(
                    "[1-2] 디스크립터 벡터 저장: HOG %sD, HSV %sD",
                    hog_desc.size,
                    color_desc.size,
                )

                logger.info("[1-2] patch_192_24: gradient/hist 결과 저장 완료")

            except Exception as e:
                logger.warning("[1-2] 결과 생성 실패: %s", e)

        except Exception as e:
            logger.warning("[1-1] annapurna_left_01 시각화 실패: %s", e)

    # 2-2) 최소 2쌍에 대해 patch/grad/color 매칭 + 매칭 거리 통계/CSV 저장
    use_pairs = pairs[: max(2, min(3, len(pairs)))]
    for i, (pl, pr) in enumerate(use_pairs, start=1):
        # a) 좌/우 이미지 로드
        imgL, imgR = load_color(pl), load_color(pr)
        # b) 동일한 설정으로 패치 추출
        patchesL, kpsL = FeatureDescriptors.make_patches(imgL, patch_w=32, patch_h=32, stride=24)
        patchesR, kpsR = FeatureDescriptors.make_patches(imgR, patch_w=32, patch_h=32, stride=24)

        # c) 세 가지 디스크립터에 대해 반복 처리
        for kind in ("patch", "grad", "color"):
            # c-1) 디스크립터 계산
            descL = FeatureDescriptors.compute_descriptors(patchesL, kind=kind)
            descR = FeatureDescriptors.compute_descriptors(patchesR, kind=kind)
            # c-2) BF 매칭 및 시각화
            out_img = paths.result_dir / f"p1_3_{kind}_match_{i:02d}.png"
            matches, _ = FeatureMatching.bfmatch_and_draw(
                imgL,
                kpsL,
                descL,
                imgR,
                kpsR,
                descR,
                out_img,
                norm_type=cv2.NORM_L2,
                do_ratio=False,
                topk=200,
            )
            # c-3) 매칭 통계 저장(txt + csv)
            stats = FeatureMatching.match_distance_stats(matches)
            txt_lines = [
                f"pair_index={i}",
                f"descriptor={kind}",
                f"num_matches={stats['n']}",
                f"dist_mean={stats['mean']:.4f}",
                f"dist_median={stats['median']:.4f}",
                f"dist_min={stats['min']:.4f}",
                f"dist_p90={stats['p90']:.4f}",
                f"dist_max={stats['max']:.4f}",
            ]
            save_stats_txt(paths.result_dir / f"p1_3_{kind}_stats_{i:02d}.txt", txt_lines)
            save_csv(
                paths.result_dir / f"p1_3_{kind}_matches_{i:02d}.csv",
                header=["rank", "distance", "queryIdx", "trainIdx"],
                rows=FeatureMatching.matches_to_rows(matches),
            )
            logger.info("[1-3] %s 매칭 저장: %s, 통계 저장", kind, out_img.name)

        # d) SIFT 매칭 및 통계(1-4)
        out = paths.result_dir / f"p1_4_sift_match_{i:02d}.png"
        k1, d1, k2, d2, matches_sift, _ = FeatureMatching.sift_detect_and_match(imgL, imgR, out, ratio=0.75, topk=500)
        stats = FeatureMatching.match_distance_stats(matches_sift)
        save_stats_txt(
            paths.result_dir / f"p1_4_sift_stats_{i:02d}.txt",
            [
                f"pair_index={i}",
                "descriptor=SIFT",
                f"num_matches={stats['n']}",
                f"dist_mean={stats['mean']:.4f}",
                f"dist_median={stats['median']:.4f}",
                f"dist_min={stats['min']:.4f}",
                f"dist_p90={stats['p90']:.4f}",
                f"dist_max={stats['max']:.4f}",
            ],
        )
        save_csv(
            paths.result_dir / f"p1_4_sift_matches_{i:02d}.csv",
            header=["rank", "distance", "queryIdx", "trainIdx"],
            rows=FeatureMatching.matches_to_rows(matches_sift),
        )
        logger.info("[1-4] SIFT 매칭/통계 저장: %s", out.name)

    # ----------------------------------------------------------------------------------
    # 3) 문제 2 — H(SVD/RANSAC) 추정 비교 + 스티칭, 통계 저장
    # ----------------------------------------------------------------------------------
    use_pairs_p2 = pairs[: max(3, min(5, len(pairs)))]
    for i, (pl, pr) in enumerate(use_pairs_p2, start=1):
        # a) 입력 로드
        imgL, imgR = load_color(pl), load_color(pr)

        # b) SIFT 매칭(공통 포인트 집합 확보)
        outm = paths.result_dir / f"p2_sift_match_{i:02d}.png"
        k1, d1, k2, d2, matches, _ = FeatureMatching.sift_detect_and_match(imgL, imgR, outm, ratio=0.75, topk=800)
        src, dst = FeatureMatching.pts_from_matches(k1, k2, matches)

        # c) 2-1 SVD 호모그래피 추정 → 스티칭 → 통계 저장
        H_svd, _ = HomographyToolkit.find_homography_svd(src, dst)
        if H_svd is not None:
            pano_svd = HomographyToolkit.stitch_two(imgL, imgR, H_svd)
            save_image(paths.result_dir / f"p2_1_stitch_svd_{i:02d}.png", pano_svd)
            stats_svd = HomographyToolkit.homography_stats(H_svd, src, dst, thresh=3.0)
            save_stats_txt(
                paths.result_dir / f"p2_1_stats_svd_{i:02d}.txt",
                [
                    f"pair_index={i}",
                    "method=SVD",
                    f"n={stats_svd['n']}",
                    f"inliers@3px={stats_svd['inliers']} ({stats_svd['inlier_ratio']:.3f})",
                    f"err_mean={stats_svd['mean_err']:.4f}",
                    f"err_median={stats_svd['median_err']:.4f}",
                    f"err_p90={stats_svd['p90_err']:.4f}",
                ],
            )
            logger.info("[2-1] SVD H 및 스티칭/통계 저장")

        # d) 2-2 RANSAC 호모그래피 추정 → 스티칭 → 통계/인라이어 시각화 저장
        H_ransac, mask_r = HomographyToolkit.find_homography_ransac(src, dst, ransacReprojThreshold=3.0, confidence=0.995)
        if H_ransac is not None:
            pano_r = HomographyToolkit.stitch_two(imgL, imgR, H_ransac)
            save_image(paths.result_dir / f"p2_2_stitch_ransac_{i:02d}.png", pano_r)
            stats_rans = HomographyToolkit.homography_stats(H_ransac, src, dst, thresh=3.0)
            save_stats_txt(
                paths.result_dir / f"p2_2_stats_ransac_{i:02d}.txt",
                [
                    f"pair_index={i}",
                    "method=RANSAC(cv2)",
                    f"n={stats_rans['n']}",
                    f"inliers@3px={stats_rans['inliers']} ({stats_rans['inlier_ratio']:.3f})",
                    f"err_mean={stats_rans['mean_err']:.4f}",
                    f"err_median={stats_rans['median_err']:.4f}",
                    f"err_p90={stats_rans['p90_err']:.4f}",
                ],
            )
            if mask_r is not None:
                inlier_matches = [m for m, ok in zip(matches, mask_r.ravel().tolist()) if ok]
                vis_in = cv2.drawMatches(
                    imgL,
                    k1,
                    imgR,
                    k2,
                    inlier_matches[:300],
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                save_image(paths.result_dir / f"p2_2_inliers_{i:02d}.png", vis_in)
            logger.info("[2-2] RANSAC H 및 스티칭/통계 저장")

    # ----------------------------------------------------------------------------------
    # 4) 문제 3 — Custom RANSAC vs OpenCV RANSAC 비교 + 스티칭
    # ----------------------------------------------------------------------------------
    use_pairs_p3 = pairs[: max(2, min(4, len(pairs)))]
    for i, (pl, pr) in enumerate(use_pairs_p3, start=1):
        # a) 공통: SIFT 매칭 확보
        imgL, imgR = load_color(pl), load_color(pr)
        outm = paths.result_dir / f"p3_sift_match_{i:02d}.png"
        k1, d1, k2, d2, matches, _ = FeatureMatching.sift_detect_and_match(imgL, imgR, outm, ratio=0.75, topk=800)
        src, dst = FeatureMatching.pts_from_matches(k1, k2, matches)

        # b) 3-1 Custom RANSAC → 스티칭/통계/인라이어 시각화
        try:
            H_c, mask_c = HomographyToolkit.ransac_homography_custom(
                src,
                dst,
                max_iters=4000,
                thresh=3.0,
                confidence=0.997,
                do_normalize=True,
                seed=0,
            )
            pano_c = HomographyToolkit.stitch_two(imgL, imgR, H_c)
            save_image(paths.result_dir / f"p3_1_stitch_custom_ransac_{i:02d}.png", pano_c)
            stats_c = HomographyToolkit.homography_stats(H_c, src, dst, thresh=3.0)
            save_stats_txt(
                paths.result_dir / f"p3_1_stats_custom_{i:02d}.txt",
                [
                    f"pair_index={i}",
                    "method=RANSAC(custom)",
                    f"n={stats_c['n']}",
                    f"inliers@3px={stats_c['inliers']} ({stats_c['inlier_ratio']:.3f})",
                    f"err_mean={stats_c['mean_err']:.4f}",
                    f"err_median={stats_c['median_err']:.4f}",
                    f"err_p90={stats_c['p90_err']:.4f}",
                ],
            )
            inlier_matches = [m for m, ok in zip(matches, mask_c.ravel().tolist()) if ok]
            vis_in = cv2.drawMatches(
                imgL,
                k1,
                imgR,
                k2,
                inlier_matches[:300],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            save_image(paths.result_dir / f"p3_1_inliers_custom_{i:02d}.png", vis_in)
            logger.info("[3-1] Custom RANSAC H 및 스티칭/통계 저장")
        except Exception as e:
            logger.warning("[3-1] Custom RANSAC 실패: %s", e)
            continue

        # c) 3-2 OpenCV RANSAC → 스티칭/통계(비교 기준)
        H_cv, mask_cv = HomographyToolkit.find_homography_ransac(src, dst, ransacReprojThreshold=3.0, confidence=0.995)
        if H_cv is not None:
            pano_cv = HomographyToolkit.stitch_two(imgL, imgR, H_cv)
            save_image(paths.result_dir / f"p3_2_stitch_cv_ransac_{i:02d}.png", pano_cv)
            stats_cv = HomographyToolkit.homography_stats(H_cv, src, dst, thresh=3.0)
            save_stats_txt(
                paths.result_dir / f"p3_2_stats_cv_{i:02d}.txt",
                [
                    f"pair_index={i}",
                    "method=RANSAC(cv2)",
                    f"n={stats_cv['n']}",
                    f"inliers@3px={stats_cv['inliers']} ({stats_cv['inlier_ratio']:.3f})",
                    f"err_mean={stats_cv['mean_err']:.4f}",
                    f"err_median={stats_cv['median_err']:.4f}",
                    f"err_p90={stats_cv['p90_err']:.4f}",
                ],
            )
            logger.info("[3-2] OpenCV RANSAC 비교 스티칭/통계 저장")

    logger.info("===== HW2 실행 종료 =====")

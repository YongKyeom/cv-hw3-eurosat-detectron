# type: ignore
# -*- coding: utf-8 -*-
"""
- 입력: data/lenna.png
- 출력: result/*.png
- 의존성: opencv-python, numpy

문제
    1) 이동/회전/아핀/원근 변환 (float32 행렬, warpAffine/warpPerspective)
    2) Gaussian/Sobel/Laplacian, S&P 노이즈, Gaussian vs Median 비교(PSNR)
    3) 보간 업/다운샘플링, Gaussian/Laplacian Pyramid, Laplacian 복원
    4) Numpy로 Median Blur 직접 구현 (cv2.medianBlur와 비교)
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

current_path = os.path.dirname(__file__)
sys.path.append(current_path)

from utils.logger import get_logger


@dataclass
class Paths:
    """
    프로젝트 경로 모음
    """

    log_dir: Path
    data_dir: Path
    result_dir: Path

    @staticmethod
    def from_root(root: str | Path) -> "Paths":
        """
        루트 경로로부터 data/result 경로 딕셔너리 생성

        Args:
            root (str | Path): 프로젝트 루트 경로.

        Returns:
            Path: logger_dir, data_dir, result_dir 경로

        """
        root = Path(root).resolve()

        # Set directory
        log = root / "log"
        data = root / "data"
        result = root / "result"

        # Create directory
        log.mkdir(parents=True, exist_ok=True)
        result.mkdir(parents=True, exist_ok=True)

        return Paths(log_dir=log, data_dir=data, result_dir=result)


def load_image(path: Path) -> np.ndarray:
    """
    BGR 컬러 이미지 로드

    Args:
        path (Path): 이미지 경로.

    Returns:
        np.ndarray: BGR 컬러 이미지.

    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {path}")

    return img


def save_image(path: Path, img: np.ndarray) -> None:
    """
    이미지 저장 (경로 자동 생성)

    Args:
        path (Path): 저장 경로.
        img (np.ndarray): 저장할 이미지.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

    return None


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    BGR -> Gray 변환 (이미 Gray면 그대로 반환)

    Args:
        img (np.ndarray): 입력 이미지.

    Returns:
        np.ndarray: Grayscale 이미지.

    """
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def psnr(
    ref: np.ndarray,
    test: np.ndarray,
    max_val: float = 255.0,
) -> float:
    """
    PSNR(Peak Signal-to-Noise Ratio) 계산

    Args:
        ref (np.ndarray): 기준 이미지.
        test (np.ndarray): 비교할 이미지.
        max_val (float): 최대 픽셀값. 기본=255.

    Returns:
        float: PSNR(dB) 값. MSE=0이면 inf 반환.

    """
    ref64 = ref.astype(np.float64)
    test64 = test.astype(np.float64)
    mse = np.mean((ref64 - test64) ** 2)
    if mse == 0:
        return float("inf")

    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


class CVOperator:
    """
    이미지 처리 기능 모듈
    """

    @staticmethod
    def translate(
        image: np.ndarray,
        tx: float,
        ty: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        평행이동: (x,y) -> (x+tx, y+ty)

        Args:
            image (np.ndarray): 입력 BGR 이미지.
            tx (float): x축 이동.
            ty (float): y축 이동.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (변환 이미지, 2x3 행렬)

        """
        h, w = image.shape[:2]
        M = np.array(
            [
                [1.0, 0.0, float(tx)],
                [0.0, 1.0, float(ty)],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(image, M, (w, h))

        return warped, M

    @staticmethod
    def rotate(
        image: np.ndarray,
        angle_deg: float,
        scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        중심 회전(+스케일): (x',y') = R(x-cx, y-cy) + (cx,cy)

        Args:
            image (np.ndarray): 입력 BGR 이미지.
            angle_deg (float): 회전 각도(도).
            scale (float): 스케일.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (변환 이미지, 2x3 행렬)

        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale)).astype(np.float32)
        warped = cv2.warpAffine(image, M, (w, h))

        return warped, M

    @staticmethod
    def affine(
        image: np.ndarray,
        src3: np.ndarray,
        dst3: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        아핀변환(3점 매칭): 선형 + 평행이동. 직선/평행성 보존.

        Args:
            image (np.ndarray): 입력 BGR 이미지.
            src3 (np.ndarray): 원본 3점 (3x2).
            dst3 (np.ndarray): 대상 3점 (3x2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (변환 이미지, 2x3 행렬)

        """
        h, w = image.shape[:2]
        M = cv2.getAffineTransform(src3.astype(np.float32), dst3.astype(np.float32)).astype(np.float32)
        warped = cv2.warpAffine(image, M, (w, h))

        return warped, M

    @staticmethod
    def perspective(
        image: np.ndarray,
        src4: np.ndarray,
        dst4: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        원근변환(4점 매칭): 3x3 호모그래피. 직선은 유지, 평행성은 보존 X.

        Args:
            image (np.ndarray): 입력 BGR 이미지.
            src4 (np.ndarray): 원본 4점 (4x2).
            dst4 (np.ndarray): 대상 4점 (4x2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (변환 이미지, 3x3 행렬)

        """
        h, w = image.shape[:2]
        H = cv2.getPerspectiveTransform(src4.astype(np.float32), dst4.astype(np.float32)).astype(np.float32)
        warped = cv2.warpPerspective(image, H, (w, h))

        return warped, H

    @staticmethod
    def gaussian(
        image: np.ndarray,
        ksize: int = 5,
        sigma: float = 1.2,
    ) -> np.ndarray:
        """
        2D 가우시안 De-noising 필터

        Args:
            image (np.ndarray): 입력 이미지.
            ksize (int): 커널 크기 (홀수).
            sigma (float): 시그마 값.

        Returns:
            np.ndarray: 블러 처리된 이미지.

        """
        k = int(ksize)
        assert k % 2 == 1 and k >= 3

        return cv2.GaussianBlur(image, (k, k), float(sigma))

    @staticmethod
    def sobel_mag(
        image: np.ndarray,
        ksize: int = 3,
    ) -> np.ndarray:
        """
        Sobel x/y 미분 -> 크기 맵(0~255)

        Args:
            image (np.ndarray): 입력 이미지.
            ksize (int): 커널 크기.

        Returns:
            np.ndarray: 소벨 크기 영상 (0~255, uint8).

        """
        gray = to_gray(image)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=int(ksize))
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=int(ksize))
        mag = np.sqrt(sx**2 + sy**2)
        mag = np.clip(mag / (mag.max() + 1e-12) * 255.0, 0, 255).astype(np.uint8)

        return mag

    @staticmethod
    def laplacian(
        image: np.ndarray,
        ksize: int = 3,
    ) -> np.ndarray:
        """
        2차 미분 기반 라플라시안 -> 0~255 시각화

        Args:
            image (np.ndarray): 입력 이미지.
            ksize (int): 커널 크기.

        Returns:
            np.ndarray: 라플라시안 결과 (정규화된 8bit).

        """
        gray = to_gray(image)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=int(ksize))
        lap_disp = ((lap - lap.min()) / (np.ptp(lap) + 1e-12) * 255.0).astype(np.uint8)

        return lap_disp

    @staticmethod
    def add_salt_pepper_noise(
        image: np.ndarray,
        prob: float = 0.08,
    ) -> np.ndarray:
        """
        Salt & Pepper 노이즈 추가
        확률 p: salt(255) / pepper(0) 치환. Gray 기준.

        Args:
            image (np.ndarray): 입력 이미지.
            prob (float): 노이즈 확률(0~1).

        Returns:
            np.ndarray: 노이즈가 추가된 이미지.

        """
        np.random.seed(0)

        out = image.copy()
        H, W = out.shape[:2]
        rnd = np.random.rand(H, W)
        salt = rnd < (prob / 2)
        pepper = rnd > (1 - prob / 2)

        if out.ndim == 2:
            out[salt] = 255
            out[pepper] = 0
        else:
            out[salt] = [255, 255, 255]
            out[pepper] = [0, 0, 0]

        return out

    @staticmethod
    def median_numpy(
        image: np.ndarray,
        ksize: int = 5,
        border: str = "replicate",
    ) -> np.ndarray:
        """
        NumPy 기반 Median Blur. Gray 입력 가정(채널별 분리 적용 가능).

        Args:
            image (np.ndarray): 입력 이미지.
            ksize (int): 커널 크기 (홀수).
            border (str): 경계 처리 모드 ('replicate'|'reflect'|'symmetric').

        Returns:
            np.ndarray: 블러 처리된 이미지.

        """
        assert ksize % 2 == 1 and ksize >= 3, "ksize는 3 이상 홀수"

        pad = ksize // 2
        pad_mode = {
            "replicate": "edge",
            "reflect": "reflect",
            "symmetric": "symmetric",
        }

        if image.ndim == 2:
            img_p = np.pad(image, ((pad, pad), (pad, pad)), mode=pad_mode[border])
            out = np.empty_like(image)
            H, W = image.shape
            for y in range(H):
                for x in range(W):
                    win = img_p[y : y + ksize, x : x + ksize].reshape(-1)
                    out[y, x] = np.median(win)

            return out

        chans = cv2.split(image)
        chans_blur = [CVOperator.median_numpy(c, ksize, border=border) for c in chans]

        return cv2.merge(chans_blur)

    @staticmethod
    def resize_once(
        image: np.ndarray,
        scale: float,
        interpolation: int,
    ) -> np.ndarray:
        """
        배율 리사이즈

        Args:
            image (np.ndarray): 입력 이미지.
            scale (float): 배율.
            interpolation (int): 보간 방식 (cv2 상수).

        Returns:
            np.ndarray: 리사이즈된 이미지.

        """
        h, w = image.shape[:2]
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))

        return cv2.resize(image, new_size, interpolation=interpolation)

    @staticmethod
    def gaussian_pyramid(
        image: np.ndarray,
        levels: int = 3,
    ) -> List[np.ndarray]:
        """
        Gaussian Pyramid 생성

        Args:
            image (np.ndarray): 입력 이미지.
            levels (int): 단계 수.

        Returns:
            List[np.ndarray]: 각 단계 이미지 리스트.

        """
        pyr = [image]
        cur = image
        for _ in range(levels):
            cur = cv2.pyrDown(cur)  # 가우시안 블러 + 1/2 축소
            pyr.append(cur)

        return pyr

    @staticmethod
    def laplacian_pyramid(gaussians: Sequence[np.ndarray]) -> List[np.ndarray]:
        """
        Laplacian Pyramid 생성

        Args:
            gaussians (Sequence[np.ndarray]): Gaussian Pyramid.

        Returns:
            List[np.ndarray]: Laplacian Pyramid.

        """
        laps: List[np.ndarray] = []
        for i in range(len(gaussians) - 1):
            g = gaussians[i].astype(np.float32)
            up = cv2.pyrUp(gaussians[i + 1], dstsize=(g.shape[1], g.shape[0])).astype(np.float32)
            laps.append(g - up)  # 고주파(세부)만 추출

        return laps

    @staticmethod
    def reconstruct_from_laplacian(
        laps: Sequence[np.ndarray],
        last_gaussian: np.ndarray,
    ) -> np.ndarray:
        """
        Laplacian Pyramid 복원

        Args:
            laps (Sequence[np.ndarray]): Laplacian Pyramid.
            last_gaussian (np.ndarray): 가장 작은 해상도의 Gaussian.

        Returns:
            np.ndarray: 복원된 이미지.

        """
        cur = last_gaussian.astype(np.float32)
        for i in range(len(laps) - 1, -1, -1):
            cur = (
                cv2.pyrUp(
                    cur,
                    dstsize=(
                        laps[i].shape[1],
                        laps[i].shape[0],
                    ),
                ).astype(np.float32)
                + laps[i]
            )
        cur = np.clip(np.rint(cur), 0, 255).astype(np.uint8)  # 반올림(np.rint) + 클립(min/max 제한)

        return cur


if __name__ == "__main__":
    ################################################################################################################
    # NOTE: Logger/Input/Output Path 지정
    project_root = Path(os.path.join(os.path.dirname(__file__), "..")).resolve()
    paths = Paths.from_root(root=project_root)
    log_dir: Path = paths.log_dir
    data_dir: Path = paths.data_dir
    result_dir: Path = paths.result_dir

    # NOTE: Set Logger
    logger = get_logger(log_dir=log_dir, name="hw01")

    # NOTE: Input data load
    src_path = f"{data_dir}/lenna.png"
    src = load_image(path=src_path)
    h, w = src.shape[:2]

    ################################################################################################################
    # NOTE: 문제 1. 변환
    trans_img, M_t = CVOperator.translate(src, 40, 30)
    rot_img, M_r = CVOperator.rotate(src, 20, 1.0)
    aff_img, M_a = CVOperator.affine(
        src,
        np.float32([[0, 0], [w - 1, 0], [0, h - 1]]),
        np.float32([[0, 40], [w - 60, 20], [40, h - 40]]),
    )
    per_img, H_p = CVOperator.perspective(
        src,
        np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
        np.float32([[40, 40], [w - 40, 0], [w - 60, h - 1], [20, h - 40]]),
    )
    save_image(result_dir / "p1_translation.png", trans_img)
    save_image(result_dir / "p1_rotation.png", rot_img)
    save_image(result_dir / "p1_affine.png", aff_img)
    save_image(result_dir / "p1_perspective.png", per_img)

    logger.info("[문제1] 행렬")
    logger.info("- Translation:\n%s", M_t)
    logger.info("- Rotation:\n%s", M_r)
    logger.info("- Affine:\n%s", M_a)
    logger.info("- Perspective:\n%s", H_p)

    ################################################################################################################
    # NOTE: 문제 2. 필터 & 노이즈
    gimg = CVOperator.gaussian(src, 5, 1.2)
    sob = CVOperator.sobel_mag(src, 3)
    lap = CVOperator.laplacian(src, 3)
    save_image(result_dir / "p2_gaussian.png", gimg)
    save_image(result_dir / "p2_sobel_mag.png", sob)
    save_image(result_dir / "p2_laplacian.png", lap)

    sp = CVOperator.add_salt_pepper_noise(src, 0.08)
    sp_g = CVOperator.gaussian(sp, 5, 1.2)
    sp_m = cv2.medianBlur(sp, 5)
    save_image(result_dir / "p2_sp.png", sp)
    save_image(result_dir / "p2_sp_gaussian.png", sp_g)
    save_image(result_dir / "p2_sp_median.png", sp_m)

    logger.info("[문제2] PSNR(원본 기준)")
    logger.info("- S&P : %.2f dB" % psnr(src, sp))
    logger.info("- S&P+Gaussian : %.2f dB" % psnr(src, sp_g))
    logger.info("- S&P+Median   : %.2f dB" % psnr(src, sp_m))

    ################################################################################################################
    # NOTE: 문제 3. 리사이즈 & 피라미드
    interps: Dict[str, int] = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    for name, flag in interps.items():
        down = CVOperator.resize_once(src, 0.5, flag)
        up = CVOperator.resize_once(down, 2.0, flag)
        save_image(result_dir / f"p3_resize_{name}_down.png", down)
        save_image(result_dir / f"p3_resize_{name}_up.png", up)

        logger.info(f"[문제3-1] {name:8s} 다운 -> 업 PSNR: {psnr(src, up):.2f} dB")

    gs = CVOperator.gaussian_pyramid(src, levels=3)  # G0..G3
    for i, g in enumerate(gs):
        save_image(result_dir / f"p3_gauss_g{i}.png", g)
    laps = CVOperator.laplacian_pyramid(gs)
    for i, l in enumerate(laps):
        vis = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        save_image(result_dir / f"p3_lap_l{i}.png", vis)
    recon = CVOperator.reconstruct_from_laplacian(laps, gs[-1])
    save_image(result_dir / "p3_recon.png", recon)
    p_psnr = psnr(src, recon)

    logger.info("[문제3-3] Laplacian 복원 PSNR: %.2f dB" % p_psnr)

    ################################################################################################################
    # NOTE: 문제 4. Median 직접구현
    sp2 = CVOperator.add_salt_pepper_noise(src, 0.12)
    np_med = CVOperator.median_numpy(sp2, 5)
    cv_med = cv2.medianBlur(sp2, 5)
    save_image(result_dir / "p4_sp.png", sp2)
    save_image(result_dir / "p4_np_median.png", np_med)
    save_image(result_dir / "p4_cv_median.png", cv_med)

    logger.info("[문제4] NumPy vs OpenCV Median PSNR: %.2f dB" % psnr(np_med, cv_med))
    logger.info("[문제4] 동일성 체크: %s", np.array_equal(np_med, cv_med))

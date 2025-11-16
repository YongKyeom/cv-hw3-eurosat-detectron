from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np


@dataclass
class BoFEncoder:
    """
    Bag-of-Features 인코더.

    Codebook(KMeans)의 transform(descriptors) 결과인
    distance matrix(N, K)를 받아 BoF histogram(K,)을 생성한다.

    Attributes:
        k (int): 시각 단어(centroid) 개수.
        normalize (Literal["l1", "l2", None]): BoW 정규화 방식.
        mode (Literal["hard", "soft"]):
            - "hard": argmin-based hard assignment
            - "soft": distance 기반 soft assignment
        soft_sigma (float): soft assignment에서 거리 → weight 변환 시 사용되는 sigma.
    """

    k: int
    normalize: Literal["l1", "l2", None] = "l1"
    mode: Literal["hard", "soft"] = "hard"
    soft_sigma: float = 20.0

    # -----------------------------------------------------------------------
    def encode(self, distances: np.ndarray) -> np.ndarray:
        """
        거리 행렬(distances)로부터 단일 이미지의 Bag-of-Features 벡터를 생성.

        Args:
            distances (np.ndarray):
                (N, K) shape.
                KMeans.transform(descriptors) 결과와 동일한 구조.

        Returns:
            np.ndarray:
                (K,) float32 histogram vector.
        """
        if distances.ndim != 2:
            raise ValueError("distances must be 2D array with shape (N, K).")

        if distances.shape[1] != self.k:
            raise ValueError(f"Expected K={self.k}, but got {distances.shape[1]}.")

        # Hard assignment ----------------------------------------------------
        if self.mode == "hard":
            # 각 feature의 최솟값 centroid index 선택
            cluster_ids = np.argmin(distances, axis=1)

            bow = np.zeros((self.k,), dtype=np.float32)
            for c in cluster_ids:
                bow[int(c)] += 1.0

        # Soft assignment ----------------------------------------------------
        elif self.mode == "soft":
            # 거리 → 가중치 변환: exp(-d / sigma)
            w = np.exp(-distances / float(self.soft_sigma))  # (N, K)

            # 각 feature의 weight sum→ image-level histogram
            bow = w.sum(axis=0).astype(np.float32)

        else:
            raise ValueError(f"Unsupported encoding mode: {self.mode}")

        # Normalization ------------------------------------------------------
        bow = self._normalize(bow)

        return bow

    # -----------------------------------------------------------------------
    def encode_batch(self, batch_distances: Sequence[np.ndarray]) -> np.ndarray:
        """
        다수의 distance matrix를 일괄 BoF histogram으로 변환한다.

        Args:
            batch_distances (Sequence[np.ndarray]): 길이 M 리스트,
                각 요소는 (Ni, K) 거리 행렬.

        Returns:
            np.ndarray: (M, K) float32 BoF 행렬.
        """
        bows: List[np.ndarray] = []
        for d in batch_distances:
            bows.append(self.encode(d))

        return np.vstack(bows).astype(np.float32)

    # -----------------------------------------------------------------------
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        벡터 정규화(L1 또는 L2).

        Args:
            vec (np.ndarray): (K,) 벡터.

        Returns:
            np.ndarray: 정규화된 벡터.
        """
        if self.normalize is None:
            return vec

        v = vec.astype(np.float32)

        if self.normalize == "l1":
            s = float(v.sum()) + 1e-6
            return v / s

        if self.normalize == "l2":
            n = float(np.linalg.norm(v)) + 1e-6
            return v / n

        raise ValueError(f"Unsupported normalize={self.normalize}")

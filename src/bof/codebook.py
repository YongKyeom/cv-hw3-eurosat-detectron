# bof/codebook.py
# -*- coding: utf-8 -*-

"""
Bag-of-Features Codebook 모듈.

- KMeans 기반 시각 단어(visual words) codebook 생성.
- train descriptors → flatten → KMeans → centroids 생성.
- 각 feature를 가까운 centroid에 매핑하기 위해 transform() 사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class VisualCodebook:
    """
    KMeans 기반 Visual Codebook (Bag-of-Features의 핵심 구성 요소)

    Attributes:
        k (int): 시각 단어(centroids) 개수.
        kmeans (KMeans): sklearn KMeans 모델.
    """

    k: int
    kmeans: Optional[KMeans] = None

    # ----------------------------------------------------------------------
    # Codebook 생성
    # ----------------------------------------------------------------------
    def fit(self, descriptors: np.ndarray) -> None:
        """
        KMeans 군집화를 통해 시각 단어 codebook을 생성한다.

        Args:
            descriptors (np.ndarray):
                모든 이미지의 descriptor를 평탄화하여 만든 (N, D) 배열.

        Notes:
            - 대규모 데이터에서는 KMeans(n_init='auto')가 안정적.
            - whiten 처리(pca whitening)는 fit 전에 외부에서 수행해야 함.
        """
        if descriptors.ndim != 2:
            raise ValueError("descriptors must be 2D array: shape (N, D)")

        self.kmeans = KMeans(
            n_clusters=self.k,
            verbose=False,
            n_init="auto",
            random_state=2025,
        )
        self.kmeans.fit(descriptors)

    # ----------------------------------------------------------------------
    # Codebook → 각 descriptor의 cluster index 할당
    # ----------------------------------------------------------------------
    def assign(self, descriptors: np.ndarray) -> np.ndarray:
        """
        각 descriptor를 가장 가까운 centroid에 할당하여
        군집 index를 반환한다.

        Args:
            descriptors (np.ndarray): (N, D) descriptor 행렬.

        Returns:
            np.ndarray: (N,) cluster index.
        """
        if self.kmeans is None:
            raise RuntimeError("Codebook is not trained. Call fit() first.")

        return self.kmeans.predict(descriptors)

    # ----------------------------------------------------------------------
    # descriptor → distances(KMeans.transform), cluster index, quantization
    # ----------------------------------------------------------------------
    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """
        각 descriptor에 대해 centroid와의 거리 벡터를 반환한다.

        Args:
            descriptors (np.ndarray): (N, D)

        Returns:
            np.ndarray: (N, K) distance matrix
        """
        if self.kmeans is None:
            raise RuntimeError("Codebook is not trained. Call fit() first.")

        return self.kmeans.transform(descriptors)

    # ----------------------------------------------------------------------
    # convenience: assign + histogram 생성
    # ----------------------------------------------------------------------
    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """
        단일 이미지의 descriptor 집합을 Bag-of-Features 벡터(히스토그램)로 인코딩한다.

        Args:
            descriptors (np.ndarray): (N, D) descriptor 행렬.

        Returns:
            np.ndarray: (K,) float32 Bag-of-Words histogram.
        """
        # 가장 가까운 시각 단어 id 추출
        cluster_ids = self.assign(descriptors)  # shape: (N,)

        # BoF histogram 생성
        hist = np.zeros((self.k,), dtype=np.float32)
        for cid in cluster_ids:
            hist[int(cid)] += 1.0

        # L1 정규화
        denom = float(hist.sum()) + 1e-6
        hist /= denom

        return hist

    # ----------------------------------------------------------------------
    # 여러 이미지에 대해 encode batch 처리
    # ----------------------------------------------------------------------
    def encode_batch(self, feature_list: list[np.ndarray]) -> np.ndarray:
        """
        여러 이미지의 descriptor 목록을 받아 일괄적으로 BoF 벡터를 생성한다.

        Args:
            feature_list (list[np.ndarray]):
                각 요소는 (N_i, D)의 descriptor 행렬.

        Returns:
            np.ndarray: (M, K) float32 BoF 행렬
                        M = 이미지 수
        """
        if self.kmeans is None:
            raise RuntimeError("Codebook is not trained. Call fit() first.")

        bows: list[np.ndarray] = []
        for desc in feature_list:
            bows.append(self.encode(desc))

        return np.vstack(bows).astype(np.float32)

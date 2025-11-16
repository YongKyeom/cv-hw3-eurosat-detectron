"""
모델 평가(evaluation)에서 공통적으로 사용하는 Metrics 유틸리티 모듈.

- Classical ML (SVM, RF, XGB)
- Deep Learning (MLP, CNN; PyTorch 기반)

두 경우 모두 y_true / y_pred만 있으면 동일한 평가 함수 사용 가능.

주요 기능:
    - accuracy 계산
    - confusion matrix 계산
    - classification report 생성
    - confusion matrix 시각화(Seaborn)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# 🔵 Accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """
    accuracy = 정확히 맞춘 비율

    Args:
        y_true (Iterable[int]): 정답 라벨
        y_pred (Iterable[int]): 예측 라벨

    Returns:
        float: accuracy (0~1)
    """
    return float(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# 🔵 Confusion Matrix
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> np.ndarray:
    """
    Confusion matrix를 numpy array로 반환.

    Args:
        y_true (Iterable[int]): 정답 라벨
        y_pred (Iterable[int]): 예측 라벨

    Returns:
        np.ndarray: (C, C) confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# 🔵 Classification Report
# ---------------------------------------------------------------------------


def compute_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    target_names: Optional[List[str]] = None,
) -> str:
    """
    precision / recall / f1-score 출력.

    Args:
        y_true (Iterable[int]): 정답 라벨
        y_pred (Iterable[int]): 예측 라벨
        target_names (List[str], optional): 클래스 이름

    Returns:
        str: classification report
    """
    return classification_report(y_true, y_pred, target_names=target_names)


# ---------------------------------------------------------------------------
# 🔵 Confusion Matrix Visualization
# ---------------------------------------------------------------------------


def save_confusion_matrix_plot(
    cm: np.ndarray,
    labels: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> None:
    """
    Confusion matrix를 Heatmap으로 저장한다.

    Args:
        cm (np.ndarray): (C, C) confusion matrix
        labels (List[str]): 클래스 이름 리스트
        save_path (Path): 저장 경로
        title (str): 그림 제목
        figsize (Tuple[int,int]): 그림 크기
        cmap (str): 색상맵
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# 🔵 통합 평가 함수 (TrainerBase에서 사용)
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """
    평가 결과를 담는 데이터 구조.

    Attributes:
        accuracy (float): 정확도(0~1)
        confusion_matrix (np.ndarray): confusion matrix
        report (str): classification report 문자열
    """

    accuracy: float
    confusion_matrix: np.ndarray
    report: str


def evaluate_classification(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Optional[List[str]] = None,
) -> MetricResult:
    """
    accuracy, confusion matrix, classification report를 한 번에 계산.

    Args:
        y_true (Iterable[int])
        y_pred (Iterable[int])
        labels (Optional[List[str]]): report에 사용할 클래스 이름

    Returns:
        MetricResult: accuracy, confusion matrix, report
    """
    acc = compute_accuracy(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    report = compute_classification_report(y_true, y_pred, target_names=labels)

    return MetricResult(
        accuracy=acc,
        confusion_matrix=cm,
        report=report,
    )

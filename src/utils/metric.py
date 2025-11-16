from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from model.metrics import evaluate_classification


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
) -> Dict[str, Any]:
    """정확도/정밀도/재현율/F1/혼동행렬을 모두 계산한다.

    Args:
        y_true (np.ndarray): 정답 라벨 배열.
        y_pred (np.ndarray): 예측 라벨 배열.
        labels (Sequence[str]): 클래스 이름 목록.

    Returns:
        Dict[str, Any]: accuracy/precision/recall/f1/confusion_matrix 를 포함한 dict.
    """
    metric_result = evaluate_classification(y_true, y_pred, list(labels))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": metric_result.accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": metric_result.confusion_matrix,
    }


def metrics_to_record(model: str, stage: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """CSV/DataFrame 저장을 위해 metric dict를 평탄화한다.

    Args:
        model (str): 모델 이름.
        stage (str): baseline/tuned 등 지표 단계명.
        stats (Dict[str, Any]): compute_classification_metrics 결과 dict.

    Returns:
        Dict[str, Any]: CSV 저장을 위한 flat dict.
    """
    cm = stats["confusion_matrix"]
    cm_list: List[List[float]] = cm.tolist() if isinstance(cm, np.ndarray) else cm
    return {
        "model": model,
        "stage": stage,
        "accuracy": stats["accuracy"],
        "precision": stats["precision"],
        "recall": stats["recall"],
        "f1": stats["f1"],
        "confusion_matrix": str(cm_list),
    }

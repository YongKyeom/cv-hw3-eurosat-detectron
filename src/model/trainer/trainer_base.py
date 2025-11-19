"""
TrainerBase 모듈

- TorchTrainer와 MLTrainer가 공통적으로 상속하는 추상적 Trainer 클래스.
- 모델 학습/평가/저장/로깅/metrics 계산 등의 기능을 공통으로 사용.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


@dataclass
class TrainerConfig:
    """
    Trainer 설정값.

    Attributes:
        save_dir (Path): 모델 저장 디렉토리
        metric (str): 평가 metric 이름 ("accuracy")
        verbose (bool): 로그 출력 여부
    """

    save_dir: Path
    metric: str = "accuracy"
    verbose: bool = True


class TrainerBase(abc.ABC):
    """
    모든 Trainer(TorchTrainer, MLTrainer)가 공통으로 가져야 할 기능을 정의하는 Base Class.

    - fit(), predict(), evaluate()는 하위 클래스에서 구현.
    - save_model(), load_model(), compute_metrics() 등은 공통 제공.
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.save_dir = config.save_dir
        self.metric = config.metric
        self.verbose = config.verbose

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger()

    # ----------------------------------------------------------------------
    # 필수 구현 메서드: 하위 클래스에서 반드시 override
    # ----------------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        모델 학습 함수.
        TorchTrainer/MLTrainer에서 각각 구현해야 한다.
        """
        pass

    @abc.abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """
        입력 X에 대한 모델의 예측을 반환.
        """
        pass

    @abc.abstractmethod
    def save_model(self, path: Path) -> None:
        """
        모델 저장 기능.
        TorchTrainer는 state_dict, MLTrainer는 pickle 형태로 저장.
        """
        pass

    @abc.abstractmethod
    def load_model(self, path: Path) -> None:
        """
        모델 로딩 기능.
        """
        pass

    # ----------------------------------------------------------------------
    # 공통 Metrics 계산
    # ----------------------------------------------------------------------
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        공통 Metric 계산 (현재는 accuracy 중심)

        Args:
            y_true (np.ndarray): 정답 라벨
            y_pred (np.ndarray): 예측 라벨

        Returns:
            Dict[str, Any]: {"accuracy": float, "confusion_matrix": ndarray}
        """
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm,
        }

    # ----------------------------------------------------------------------
    # 공통 평가 함수
    # ----------------------------------------------------------------------
    def evaluate(self, X: Any, y: np.ndarray) -> Dict[str, Any]:
        """
        모델 평가 (accuracy + confusion matrix)

        Args:
            X (Any): 입력 데이터 (PyTorch loader 혹은 numpy 배열)
            y (np.ndarray): 정답 라벨

        Returns:
            Dict[str, Any]: metric 결과 dict
        """
        y_pred = self.predict(X)
        metrics = self.compute_metrics(y, y_pred)

        return metrics

    # ----------------------------------------------------------------------
    # 모델 저장 헬퍼
    # ----------------------------------------------------------------------
    def save(self, name: str = "model") -> Path:
        """
        모델을 save_dir에 name.pt 또는 name.pkl 형태로 저장한다.

        Args:
            name (str): 파일명 (확장자는 하위 클래스가 지정)

        Returns:
            Path: 저장된 파일 경로
        """
        path = self.save_dir / name
        self.save_model(path)

        if self.verbose and self.logger:
            self.logger.info("Model saved at: %s", path)

        return path

    # ----------------------------------------------------------------------
    # 하이퍼파라미터 튜닝 인터페이스
    # ----------------------------------------------------------------------
    @abc.abstractmethod
    def hyperopt_search(self, *args, **kwargs) -> Dict[str, Any]:
        """
        HyperoptRunner를 호출하기 위한 공통 인터페이스.

        실제 구현은 model/optim/hyperopt_runner.py에서 이루어짐.
        """
        raise NotImplementedError("HyperoptRunner 호출은 각 Trainer에서 구현해야 합니다.")

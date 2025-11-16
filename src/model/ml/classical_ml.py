"""
Classical ML Model Wrapper

- SVM, RandomForest, XGBoost 를 공통 인터페이스로 다룰 수 있게 하는 래퍼 클래스들.
- Trainer(MLTrainer)가 이 모듈의 공통 interface를 기반으로
    모델 종속성 없이 동일한 방식으로 training/prediction 가능하게 설계.

지원 모델:
    - SVM (sklearn.svm.SVC)
    - RandomForest (sklearn.ensemble.RandomForestClassifier)
    - XGBoost (xgboost.XGBClassifier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from model.dl.utils import seed_everything

# ---------------------------------------------------------------------------
# 🔵 Base Interface
# ---------------------------------------------------------------------------


class BaseMLModel:
    """
    Classical ML 모델을 위한 공통 인터페이스.

    Trainer(MLTrainer)가 이 인터페이스만 보고 fit/predict 할 수 있도록 설계한다.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """모델 학습"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """클래스 예측"""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """확률 예측(SVM 일부 커널은 제공 안 함). 없으면 None 반환"""
        return None

    def get_model(self) -> Any:
        """내부 원본 모델 객체 반환"""
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        """모델의 현재 파라미터 dict"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 🔵 1. SVM Wrapper
# ---------------------------------------------------------------------------


@dataclass
class SVMModel(BaseMLModel):
    """
    SVM Classifier Wrapper

    Attributes:
        params (Dict[str, Any]): sklearn.svm.SVC 에 전달할 파라미터
    """

    params: Dict[str, Any]

    def __post_init__(self):
        # probability=True 설정 시 predict_proba 사용 가능
        default = {"kernel": "linear", "C": 1.0, "probability": True}
        merged = {**default, **self.params}
        self.model = SVC(**merged)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def get_model(self) -> Any:
        return self.model

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()


# ---------------------------------------------------------------------------
# 🔵 2. RandomForest Wrapper
# ---------------------------------------------------------------------------


@dataclass
class RFModel(BaseMLModel):
    """
    RandomForest Classifier Wrapper

    Attributes:
        params (Dict[str, Any]): RandomForestClassifier 파라미터 dict
    """

    params: Dict[str, Any]

    def __post_init__(self) -> None:
        default = {
            "n_estimators": 300,
            "max_depth": None,
            "random_state": 2025,
            "n_jobs": -1,
        }
        merged = {**default, **self.params}
        self.model = RandomForestClassifier(**merged)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return self.model.predict_proba(X)

    def get_model(self) -> Any:
        return self.model

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()


# ---------------------------------------------------------------------------
# 🔵 3. XGBoost Wrapper
# ---------------------------------------------------------------------------


@dataclass
class XGBModel(BaseMLModel):
    """
    XGBoost Classifier Wrapper

    Attributes:
        params (Dict[str, Any]): xgboost.XGBClassifier 파라미터 dict
    """

    params: Dict[str, Any]

    def __post_init__(self) -> None:
        default = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 2025,
            "tree_method": "auto",
            "eval_metric": "mlogloss",
        }

        if self.params["is_valid"] is True:
            default["early_stopping_rounds"] = 50
        self.params.pop("is_valid", None)

        merged = {**default, **self.params}
        self.model = XGBClassifier(**merged)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return self.model.predict_proba(X)

    def get_model(self) -> Any:
        return self.model

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()


# ---------------------------------------------------------------------------
# 🔵 4. Factory: 문자열로 모델 생성
# ---------------------------------------------------------------------------


def create_classical_model(
    model_type: Literal["svm", "rf", "xgb"],
    params: Optional[Dict[str, Any]] = None,
    is_valid: bool = False,
) -> BaseMLModel:
    """
    문자열로 classical ML 모델을 생성하는 팩토리 함수.

    Args:
        model_type (str): {"svm", "rf", "xgb"}
        params (dict): 모델 파라미터 dict
        is_valid (bool): Validation 여부. True인 경우, XGBoost의 early stopping 인자를 추가하기 위한 구분자

    Returns:
        BaseMLModel: SVMModel / RFModel / XGBModel 중 하나
    """
    assert model_type in ["svm", "rf", "xgb"], f"지원하지 않는 모델: {model_type}"
    params = params or {}

    # Set Random Seed
    seed_everything(2025)

    # Define model
    if model_type == "svm":
        return SVMModel(params=params)

    elif model_type == "rf":
        return RFModel(params=params)

    else:
        params["is_valid"] = is_valid
        return XGBModel(params=params)

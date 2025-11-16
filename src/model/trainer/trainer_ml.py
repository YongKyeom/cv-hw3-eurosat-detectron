"""
MLTrainer

클래식 머신러닝(SVM / RF / XGB)의 학습 및 Hyperopt 기반 파라미터 탐색을 담당.
Torch 기반 Trainer와 인터페이스를 통일하였음.

핵심 기능:
- fit(): 모델 학습
- predict(): 예측
- hyperopt_search(): HyperoptRunner를 이용해 최적 파라미터 탐색
- best params 기반 모델 재구성 및 전체 학습 데이터로 재학습
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from model.ml.classical_ml import create_classical_model
from model.trainer.trainer_base import TrainerBase, TrainerConfig


class MLTrainer(TrainerBase):
    """
    Classical ML Trainer

    Args:
        model: BaseMLModel (SVMModel, RFModel, XGBModel 래퍼 객체)
        config: TrainerConfig
    """

    def __init__(self, model, config: TrainerConfig):
        super().__init__(config)
        self.model = model

    # -------------------------------------------------------------------
    # 기본 fit / predict / save / load
    # -------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, *args, **kwargs) -> None:
        """클래식 ML 모델 학습"""
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        return self.model.predict(X)

    def save_model(self, path):
        """pickle로 모델 저장"""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        """pickle로 모델 로드"""
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    # -------------------------------------------------------------------
    # Hyperopt Search
    # -------------------------------------------------------------------

    def hyperopt_search(
        self,
        model_type: str,
        search_runner,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_evals: int = 20,
    ) -> Tuple[Any, Dict[str, Any], Any]:
        """
        HyperoptRunner를 이용해 파라미터 탐색을 수행한다.

        Args:
            model_type (str): "svm" | "rf" | "xgb"
            search_runner: HyperoptRunner instance
            X_train (np.ndarray): 학습 데이터
            y_train (np.ndarray): 학습 라벨
            X_val (np.ndarray): 검증 데이터
            y_val (np.ndarray): 검증 라벨
            max_evals (int): Hyperopt 반복 횟수

        Returns:
            best_model: 최적 파라미터로 재학습된 모델
            best_params (dict): 최적 파라미터
            logs_df (DataFrame): Hyperopt 탐색 로그
        """
        if self.logger:
            self.logger.info("[ML][%s] Hyperopt 시작", model_type.upper())

        def objective(params: Dict[str, Any]):
            # 모델 생성
            model = create_classical_model(model_type=model_type, params=params, is_valid=True)

            fit_kwargs = {}
            if model_type == "xgb":
                fit_kwargs = {
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,
                }
            model.fit(X_train, y_train, **fit_kwargs)

            # 평가
            preds = model.predict(X_val)
            acc = float((preds == y_val).mean())
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val,
                preds,
                average="macro",
                zero_division=0,
            )
            loss = 1.0 - f1

            if self.verbose:
                msg = (
                    f"[MLTrainer][{model_type.upper()}] params={params} "
                    f"acc={acc:.4f} prec={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
                )
                if self.logger:
                    self.logger.info(msg)

            return {"loss": loss, "status": "ok"}

        # Hyperopt 실행
        best_params, logs_df = search_runner.optimize(
            space_name=model_type,
            objective_fn=objective,
            max_evals=max_evals,
        )

        if model_type == "xgb":
            # n_estimators → Early Stopping으로 최적화
            best_model = create_classical_model(model_type=model_type, params=best_params, is_valid=True)
            best_model.fit(
                X_train,
                y_train,
                **{
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,
                },
            )

            try:
                best_params["n_estimators"] = max(int(best_model.model.best_iteration), 50)
                self.model = best_model
            except Exception:
                if self.logger:
                    self.logger.warning("[ML][%s] Early stopping 정보 없음", model_type.upper())

        if self.logger:
            self.logger.info("[ML][%s] Best Params=%s", model_type.upper(), best_params)

        if model_type == "xgb":
            return best_model, best_params, logs_df

        # 최적 파라미터로 모델 재생성 및 전체 학습데이터로 재학습
        best_model = create_classical_model(model_type=model_type, params=best_params, is_valid=False)
        best_model.fit(X_train, y_train)
        self.model = best_model  # Trainer 내부 모델을 업데이트함

        return best_model, best_params, logs_df

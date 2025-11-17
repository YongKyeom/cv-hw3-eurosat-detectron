"""
Hyperopt Runner (Rewritten Version)

- Hyperopt TPE 기반 파라미터 탐색 엔진
- Search Space를 파일 내부에서 통합 관리
- Trainer(MLTrainer/TorchTrainer)가 objective_fn을 넘기면 최적화 수행
- 최적 params와 trials DataFrame 반환
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, hp, space_eval, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll.base import scope

# -----------------------------------------------------------
# 1) Search Space definitions
# -----------------------------------------------------------

SEARCH_SPACES = {
    "svm": {
        "C": hp.loguniform("svm_C", -5, 5),
        "kernel": hp.choice("svm_kernel", ["linear", "rbf"]),
    },
    "rf": {
        "n_estimators": scope.int(hp.quniform("rf_estimators", 300, 1500, q=150)),
        "max_features": hp.quniform("max_features", 0.5, 1.0, q=0.1),
    },
    "xgb": {
        "max_depth": scope.int(hp.quniform("xgb_max_depth", 3, 12, q=3)),
        "learning_rate": hp.loguniform("xgb_learning_rate", -3, -1),
        "subsample": hp.quniform("xgb_subsample", 0.5, 1.0, q=0.1),
        "colsample_bytree": hp.quniform("xgb_colsample_bytree", 0.5, 1.0, q=0.1),
    },
    "mlp": {
        "num_layers": scope.int(hp.quniform("mlp_layers", 2, 4, 1)),
        "first_dim": scope.int(hp.quniform("mlp_first_dim", 128, 768, 64)),
        "width_decay": hp.uniform("mlp_width_decay", 0.3, 0.8),
        "dropout": hp.quniform("mlp_dropout", 0.1, 0.4, q=0.1),
        "lr": hp.loguniform("mlp_lr", -5, -2),
    },
    "cnn": {
        # "num_blocks": scope.int(hp.quniform("cnn_blocks", 2, 3, 1)),
        "num_blocks": hp.choice("cnn_blocks", [2]),
        "base_channels": scope.int(hp.quniform("cnn_base_channels", 8, 48, 8)),
        "channel_growth": hp.quniform("cnn_channel_growth", 1.5, 2.0, 0.1),
        # "use_batchnorm": hp.choice("cnn_use_bn", [True, False]),
        "dropout": hp.quniform("cnn_dropout", 0.1, 0.3, q=0.1),
        "lr": hp.loguniform("cnn_lr", -5, -2),
    },
    "resnet": {
        "layer1": scope.int(hp.quniform("resnet_layer1", 2, 3, 1)),
        "layer2": scope.int(hp.quniform("resnet_layer2", 2, 3, 1)),
        "layer3": scope.int(hp.quniform("resnet_layer3", 2, 3, 1)),
        "base_channels": scope.int(hp.quniform("resnet_base_channels", 8, 48, 8)),
        "lr": hp.loguniform("resnet_lr", -5, -2),
    },
    "convnext": {
        "depth1": scope.int(hp.quniform("convnext_depth1", 1, 3, 1)),
        "depth2": scope.int(hp.quniform("convnext_depth2", 1, 3, 1)),
        "depth3": scope.int(hp.quniform("convnext_depth3", 1, 3, 1)),
        "base_dim": scope.int(hp.quniform("convnext_base_dim", 32, 96, 8)),
        "dim_growth": hp.uniform("convnext_dim_growth", 1.5, 2.0),
        "lr": hp.loguniform("convnext_lr", -5, -2),
    },
}


# -----------------------------------------------------------
# 2) Runner Class
# -----------------------------------------------------------


class HyperoptRunner:
    """Hyperopt 기반 파라미터 탐색기."""

    def __init__(self, search_spaces: Dict[str, Dict[str, Any]] = SEARCH_SPACES):
        """탐색 공간을 초기화한다.

        Args:
            search_spaces (Dict[str, Dict[str, Any]]): 모델별 Hyperopt space.
        """
        self.search_spaces = search_spaces

    def optimize(
        self,
        space_name: str,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        max_evals: int = 100,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Hyperopt TPE 최적화를 실행한다.

        Args:
            space_name (str): 탐색할 space 이름 ("svm", "rf", "xgb", "mlp", "cnn", "resnet", "convnext").
            objective_fn (Callable[[Dict[str, Any]], Dict[str, Any]]): Hyperopt objective.
            max_evals (int): 반복 횟수.

        Returns:
            Tuple[Dict[str, Any], pd.DataFrame]: (최적 파라미터, 탐색 로그 DataFrame).
        """
        if space_name not in self.search_spaces:
            raise KeyError(f"Search space '{space_name}' not found")

        # Search 공간 정의
        space = self.search_spaces[space_name]

        # Execute optimizing
        trials = Trials()
        best_params_raw = fmin(
            fn=objective_fn,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(2025),
            early_stop_fn=no_progress_loss(15),
        )
        # Best hyper-paramter
        best_params = space_eval(space, best_params_raw)

        # logs_df 생성
        records: list = []
        for t in trials.trials:
            vals = t["misc"]["vals"]
            raw_params = {k: v[0] if isinstance(v, list) else v for k, v in vals.items()}
            decoded = space_eval(space, raw_params)
            result = t.get("result", {})
            loss_val = result.get("loss")
            records.append(
                {
                    "loss": loss_val,
                    "f1_score": None if loss_val is None else 1.0 - float(loss_val),
                    "status": result.get("status"),
                    **decoded,
                }
            )
        logs_df = pd.DataFrame(records)

        return best_params, logs_df

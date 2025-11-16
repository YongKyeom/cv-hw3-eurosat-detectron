from __future__ import annotations

import copy
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from model.dl.cnn import CNNClassifier, CNNConfig
from model.dl.convnext import ConvNeXtClassifier, ConvNeXtConfig
from model.dl.mlp import MLPClassifier, MLPConfig
from model.dl.resnet import ResNetClassifier, ResNetConfig
from model.dl.utils import get_device
from model.trainer.trainer_base import TrainerBase, TrainerConfig


class TorchTrainer(TrainerBase):
    """PyTorch trainer with built-in hyperopt search.

    - CNN/MLP/ResNet/ConvNeXt 등 nn.Module 학습을 담당한다.
    - Early stopping, checkpoint 저장/로드, Hyperopt 기반 탐색 기능을 포함한다.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        early_stopping_patience: int = 10,
        model_name: str = "DL",
    ):
        super().__init__(config)
        self.device = get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.early_stopping_patience = early_stopping_patience
        self.model_name = model_name

    # ------------------------------------------------------------------
    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        """단일 epoch 학습을 수행하고 평균 loss 를 반환한다."""
        self.model.train()
        total = 0

        for X, y in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total += loss.item() * X.size(0)

        return total / len(train_loader.dataset)

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """유효성 데이터에서 accuracy/precision/recall/F1을 계산한다."""
        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                out = self.model(X)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                labels.extend(y.numpy())

        preds_arr = np.asarray(preds)
        labels_arr = np.asarray(labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr,
            preds_arr,
            average="macro",
            zero_division=0,
        )
        acc = float((preds_arr == labels_arr).mean())
        return {
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader=None, num_epochs=10):
        """주어진 데이터로 모델을 학습하고 F1 기준으로 best state를 보존한다."""
        best_f1 = float("-inf")
        best_state = None
        wait = 0

        for epoch in range(num_epochs):
            _ = self._train_one_epoch(train_loader)
            if val_loader:
                metrics = self._validate(val_loader)
                if self.verbose and self.logger:
                    self.logger.info(
                        "[%s][Epoch %d] Val Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f",
                        self.model_name.upper(),
                        epoch + 1,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                    )
                if metrics["f1"] > best_f1 + 1e-4:
                    best_f1 = metrics["f1"]
                    best_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.early_stopping_patience:
                        break

        if best_state is not None:
            # Load Best 모델
            self.model.load_state_dict(best_state)

    def predict(self, loader: DataLoader) -> np.ndarray:
        """DataLoader 전체에 대해 forward를 수행하고 예측 라벨을 반환한다."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                out = self.model(X)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
        return np.array(preds)

    # Save/Load --------------------------------------------------------
    def save_model(self, path):
        """state_dict를 디스크에 저장한다."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """저장된 state_dict를 로드한다."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # ------------------------------------------------------------------
    # Hyperopt Search (DL)
    # ------------------------------------------------------------------

    def hyperopt_search(
        self,
        model_type: str,
        search_runner,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_evals: int = 100,
    ) -> Tuple[Any, Dict[str, Any], Any]:
        if self.logger:
            self.logger.info("[DL][%s] Hyperopt 시작", model_type.upper())

        def _round_channel(value: float) -> int:
            """채널 수를 8단위로 반올림하여 하드웨어 친화적인 수치로 만든다."""
            return max(16, int(round(value / 8) * 8))

        def _build_mlp_hidden(p: Dict[str, Any]) -> List[int]:
            """MLP width와 layer 수를 기반으로 hidden dimension 리스트를 생성."""
            width = float(p["first_dim"])
            hidden: List[int] = []
            for _ in range(int(p["num_layers"])):
                hidden.append(max(64, int(round(width / 16) * 16)))
                width *= float(p["width_decay"])
            return hidden

        def _build_cnn_channels(p: Dict[str, Any]) -> List[int]:
            """Conv block 수와 성장률을 기반으로 channel 배열을 생성."""
            width = float(p["base_channels"])
            growth = float(p["channel_growth"])
            channels: List[int] = []
            for _ in range(int(p["num_blocks"])):
                channels.append(_round_channel(width))
                width *= growth
            return channels

        def _build_resnet_layers(p: Dict[str, Any]) -> Tuple[int, int, int]:
            """ResNet 각 stage block 수를 튜플로 변환."""
            return (int(p["layer1"]), int(p["layer2"]), int(p["layer3"]))

        def _build_convnext_depths(p: Dict[str, Any]) -> Tuple[int, int, int]:
            """ConvNeXt stage별 깊이를 생성."""
            return (int(p["depth1"]), int(p["depth2"]), int(p["depth3"]))

        def _build_convnext_dims(p: Dict[str, Any]) -> Tuple[int, int, int]:
            """ConvNeXt stage별 channel 수를 생성."""
            width = float(p["base_dim"])
            growth = float(p["dim_growth"])
            dims: List[int] = []
            for _ in range(3):
                dims.append(_round_channel(width))
                width *= growth
            return tuple(dims)

        def _augment_params(p: Dict[str, Any]) -> Dict[str, Any]:
            """Hyperopt에서 뽑은 primitive 파라미터를 실제 설정값으로 확장한다."""
            enriched = dict(p)
            if model_type == "mlp":
                enriched["hidden"] = _build_mlp_hidden(enriched)
            elif model_type == "cnn":
                enriched["channels"] = _build_cnn_channels(enriched)
            elif model_type == "resnet":
                enriched["layers"] = _build_resnet_layers(enriched)
            elif model_type == "convnext":
                enriched["depths"] = _build_convnext_depths(enriched)
                enriched["dims"] = _build_convnext_dims(enriched)
            return enriched

        def objective(params):
            enriched_params = _augment_params(params)
            if model_type == "mlp":
                cfg = MLPConfig(
                    input_dim=32 * 32 * 3,
                    num_classes=10,
                    hidden_dims=enriched_params["hidden"],
                    dropout=enriched_params["dropout"],
                )
                model = MLPClassifier(cfg)

            elif model_type == "cnn":
                cfg = CNNConfig(
                    num_classes=10,
                    channels=enriched_params["channels"],
                    dropout=enriched_params["dropout"],
                    use_batchnorm=enriched_params.get("use_batchnorm", True),
                )
                model = CNNClassifier(cfg)

            elif model_type == "resnet":
                cfg = ResNetConfig(
                    num_classes=10,
                    layers=enriched_params["layers"],
                    base_channels=enriched_params["base_channels"],
                )
                model = ResNetClassifier(cfg)

            elif model_type == "convnext":
                cfg = ConvNeXtConfig(
                    num_classes=10,
                    depths=enriched_params["depths"],
                    dims=enriched_params["dims"],
                )
                model = ConvNeXtClassifier(cfg)

            else:
                raise ValueError(f"Unknown DL model {model_type}")

            lr = enriched_params.get("lr", 1e-3)
            trainer = TorchTrainer(
                model=model,
                config=TrainerConfig(save_dir=self.save_dir, verbose=False),
                optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                early_stopping_patience=3,
                model_name=model_type,
            )
            trainer.fit(train_loader, val_loader, num_epochs=enriched_params["epochs"])

            preds, targets = [], []
            trainer.model.eval()
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(trainer.device)
                    out = trainer.model(X)
                    pred = out.argmax(dim=1).cpu().numpy()
                    preds.extend(pred.tolist())
                    targets.extend(y.numpy().tolist())

            preds = np.asarray(preds)
            targets = np.asarray(targets)
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets,
                preds,
                average="macro",
                zero_division=0,
            )
            acc = float((preds == targets).mean())

            if self.logger:
                self.logger.info(
                    "[DL][%s] params=%s Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f",
                    model_type.upper(),
                    enriched_params,
                    acc,
                    precision,
                    recall,
                    f1,
                )

            result = {"loss": 1 - f1, "status": "ok"}

            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        best_params, logs_df = search_runner.optimize(
            space_name=model_type,
            objective_fn=objective,
            max_evals=max_evals,
        )

        # best model 재생성 + 재학습
        enriched_best = _augment_params(best_params)

        if model_type == "mlp":
            cfg = MLPConfig(
                input_dim=32 * 32 * 3,
                num_classes=10,
                hidden_dims=enriched_best["hidden"],
                dropout=enriched_best["dropout"],
            )
            best_model = MLPClassifier(cfg)

        elif model_type == "cnn":
            cfg = CNNConfig(
                num_classes=10,
                channels=enriched_best["channels"],
                dropout=enriched_best["dropout"],
                use_batchnorm=enriched_best.get("use_batchnorm", True),
            )
            best_model = CNNClassifier(cfg)

        elif model_type == "resnet":
            cfg = ResNetConfig(
                num_classes=10,
                layers=enriched_best["layers"],
                base_channels=enriched_best["base_channels"],
            )
            best_model = ResNetClassifier(cfg)

        else:
            cfg = ConvNeXtConfig(
                num_classes=10,
                depths=enriched_best["depths"],
                dims=enriched_best["dims"],
            )
            best_model = ConvNeXtClassifier(cfg)

        self.model = best_model.to(self.device)

        lr = enriched_best.get("lr", 1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.logger:
            self.logger.info("[DL][%s] Best Params=%s", model_type.upper(), enriched_best)

        self.fit(train_loader, val_loader, num_epochs=enriched_best["epochs"])

        return self.model, enriched_best, logs_df

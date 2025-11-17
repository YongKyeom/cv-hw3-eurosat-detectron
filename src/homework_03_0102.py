# type: ignore
"""HW3 문제 1/2 전용 스크립트.

문제별로 수행하는 작업 개요:

문제 1 — Bag-of-Features + Classical ML (SVM/RF/XGB)
    • 특징 추출: Scene-15 이미지를 격자형 키포인트로 분할 후 OpenCV SIFT 디스크립터 추출
    • BoF 인코딩: VisualCodebook(K-means)과 BoFEncoder로 이미지별 히스토그램 생성
    • 분류기 학습/튜닝: SVM/RF/XGB Baseline 학습 → Hyperopt 기반 파라미터 탐색
    • 결과 분석: Train/Test Confusion Matrix, Hyperopt 로그/산점도/히트맵, F1/정확도 로그


문제 2 — EuroSAT 분류 (MLP/CNN/ResNet/ConvNeXt)
    • 데이터 전처리: torchvision.transforms 로 32×32 resize 및 augmentation, Train/Val/Test 분할
    • 모델 학습: 각 모델 Baseline 학습(early stopping 포함)
    • Hyperopt: layer/width/dropout 등 세밀한 파라미터 공간 탐색 및 checkpoint 캐싱
    • 분석: Train/Test Confusion Matrix 저장, Hyperopt 시각화(산점도/히트맵/parallel), CSV로 정리

전체 실행 흐름
    1) Paths/logger/seed 초기화
    2) 문제 1 실행 → 결과(BoF 히스토그램/Confusion Matrix/Hyperopt 분석) 저장
    3) 문제 2 실행 → EuroSAT 데이터셋 로드/모델 학습/Hyperopt/시각화
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from bof.codebook import VisualCodebook
from bof.encoder import BoFEncoder
from features.descriptors import compute_sift_descriptors
from features.patch import extract_grid_patches
from model.dataset_eurosat import EuroSATConfig, load_eurosat
from model.dl.utils import build_dl_model, collect_loader_targets, seed_everything
from model.ml.classical_ml import create_classical_model
from model.optim.hyperopt_runner import HyperoptRunner
from model.trainer.trainer_base import TrainerConfig
from model.trainer.trainer_ml import MLTrainer
from model.trainer.trainer_torch import TorchTrainer
from utils.io import load_color, load_json, save_json
from utils.logger import get_logger
from utils.metric import compute_classification_metrics, metrics_to_record
from utils.paths import Paths
from utils.visualize import (
    plot_confusion_matrix,
    plot_parallel_param_coordinates,
    plot_param_heatmaps,
    plot_param_performance_curves,
    save_histogram,
)

# Main Logger
logger: Optional[logging.Logger] = None


def _extract_grid_sift(image: np.ndarray, patch_hw: int, stride: int) -> np.ndarray:
    """격자형으로 정의한 키포인트에서 OpenCV SIFT 디스크립터를 추출한다.

    Args:
        image (np.ndarray): SIFT를 적용할 BGR 이미지 (H, W, 3).
        patch_hw (int): 패치 한 변의 길이. 값이 커질수록 receptive field가 확대된다.
        stride (int): 패치 추출 간격. stride가 작을수록 더 촘촘한 키포인트가 생성된다.

    Returns:
        np.ndarray: 추출된 SIFT 디스크립터 배열 (N, 128). 주요 특징점이 없을 경우 zero vector를 반환.
    """
    _, keypoints = extract_grid_patches(image, patch_hw, patch_hw, stride)
    _, desc = compute_sift_descriptors(image, keypoints)
    if desc.size == 0:
        desc = np.zeros((1, 128), dtype=np.float32)
    return desc.astype(np.float32)


def _get_dataset_class_counts(dataset: Any, num_classes: int) -> Optional[List[int]]:
    """Subset 기반 Dataset에서 클래스별 샘플 수를 계산한다.

    Args:
        dataset: torchvision Subset 혹은 내부에 Subset이 중첩된 Dataset.
        num_classes (int): 클래스 개수.

    Returns:
        Optional[List[int]]: 클래스별 샘플 수. 계산할 수 없으면 None.
    """

    if dataset is None:
        return None

    subset = getattr(dataset, "subset", None)
    if subset is None or not hasattr(subset, "indices"):
        return None

    indices = np.asarray(subset.indices, dtype=np.int64)
    base_dataset = subset.dataset

    # 중첩 Subset 구조 일 경우 풀어서 최종 원본까지 접근
    while hasattr(base_dataset, "subset") and hasattr(base_dataset.subset, "indices"):
        parent_subset = base_dataset.subset
        parent_indices = np.asarray(parent_subset.indices, dtype=np.int64)
        indices = parent_indices[indices]
        base_dataset = parent_subset.dataset

    targets: Optional[np.ndarray] = None
    if hasattr(base_dataset, "targets"):
        targets = np.asarray(base_dataset.targets, dtype=np.int64)
    elif hasattr(base_dataset, "samples"):
        targets = np.asarray([label for _, label in getattr(base_dataset, "samples", [])], dtype=np.int64)

    if targets is None or indices.size == 0:
        return None

    counts = np.bincount(targets[indices], minlength=num_classes)
    return counts.tolist()


def _log_split_distribution(loader: DataLoader, split_name: str, num_classes: int) -> None:
    """주어진 DataLoader의 클래스 분포를 로그로 남긴다."""
    counts = _get_dataset_class_counts(loader.dataset, num_classes)
    if counts is None:
        return
    ratios = np.round(np.asarray(counts, dtype=np.float64) / max(1, sum(counts)), 4).tolist()
    logger.info("[문제2][%s] Ratio=%s", split_name, ratios)


def _evaluate_dl_split(
    trainer: TorchTrainer,
    loader: DataLoader,
    labels: List[str],
    split_name: str,
    save_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    """DL 모델의 split별 정량 지표와 Confusion Matrix를 계산한다.

    Args:
        trainer (TorchTrainer): 학습된 트레이너.
        loader (DataLoader): 평가 대상 loader.
        labels (List[str]): 클래스 이름 목록.
        split_name (str): 구분을 위한 split명 (train/test 등).
        save_dir (Path): Confusion matrix 저장 위치.
        model_name (str): 모델 이름(로그용).

    Returns:
        Dict[str, Any]: accuracy/precision/recall/F1 및 confusion matrix.
    """
    preds = trainer.predict(loader)
    y_true = collect_loader_targets(loader)
    stats = compute_classification_metrics(y_true, preds, labels)
    plot_confusion_matrix(stats["confusion_matrix"], labels, save_dir / f"{split_name}_confusion.png")
    if logger:
        logger.info(
            "[문제2][%s][%s] Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f\n%s",
            model_name.upper(),
            split_name,
            stats["accuracy"],
            stats["precision"],
            stats["recall"],
            stats["f1"],
            stats["confusion_matrix"],
        )
    return stats


# ------------------------------------------------------------------
# HW3 w/ 1, 2번 문제 Main Execute
# ------------------------------------------------------------------
if __name__ == "__main__":
    st_time = datetime.now()

    # ------------------------------------------------------------------
    # 0. 경로/로거/시드 설정
    # ------------------------------------------------------------------
    project_root = Path(os.path.join(os.path.dirname(__file__), "..")).resolve()
    paths = Paths.from_root(project_root)
    data_dir = paths.data_dir
    result_dir = paths.result_dir

    logger = get_logger(paths.log_dir, "hw03_0102")
    logger.info("Project Root: %s", project_root)
    logger.info("Result Root: %s", result_dir)

    seed_everything(2025)

    # ==============================================================================
    # 문제 1: Bag-of-Features + Classical ML
    # ==============================================================================
    # ------------------------------------------------------------------
    # 1. 문제 1 — BoF + Classical ML
    #     • Parameter 결과: Hyperopt 로그 + 최적 파라미터
    #     • 최종 스코어: Baseline/Tuned Accuracy
    #     • 시각화: Confusion Matrix(베이스/튜닝) 저장
    # ------------------------------------------------------------------
    # logger.info("==================== 문제 1: Bag-of-Features ====================")
    # scene_dir = data_dir / "SCENE-15"
    # if scene_dir.exists():
    #     train_root = scene_dir / "train"
    #     test_root = scene_dir / "test"
    #     classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    #     class_to_idx = {c: idx for idx, c in enumerate(classes)}

    #     # (1) 이미지 로드 및 라벨 생성
    #     # --- Train split 이미지/라벨/이름 수집 ---
    #     x_train, y_train, x_train_names = [], [], []
    #     for cls in classes:
    #         for img_path in sorted((train_root / cls).glob("*")):
    #             if img_path.is_file():
    #                 x_train.append(load_color(img_path))
    #                 y_train.append(class_to_idx[cls])
    #                 x_train_names.append(f"{cls}_{img_path.stem}")

    #     # --- Test split 이미지/라벨/이름 수집 ---
    #     x_test, y_test, x_test_names = [], [], []
    #     for cls in classes:
    #         for img_path in sorted((test_root / cls).glob("*")):
    #             if img_path.is_file():
    #                 x_test.append(load_color(img_path))
    #                 y_test.append(class_to_idx[cls])
    #                 x_test_names.append(f"{cls}_{img_path.stem}")

    #     # --- Train/Test 라벨 분포 ---
    #     logger.info("[문제1] Train=%d, Test=%d, Classes=%d", len(x_train), len(x_test), len(classes))
    #     train_counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=len(classes))
    #     test_counts = np.bincount(np.asarray(y_test, dtype=np.int64), minlength=len(classes))
    #     train_total = max(1, int(train_counts.sum()))
    #     test_total = max(1, int(test_counts.sum()))
    #     logger.info("[문제1] Train Ratio=%s", np.round(train_counts / train_total, 3).tolist())
    #     logger.info("[문제1] Test Ratio=%s", np.round(test_counts / test_total, 3).tolist())

    #     # (2) 격자 패치 기반 SIFT → BoF 벡터
    #     # --- 격자형 패치 위치를 활용한 SIFT 추출 (patch_hw/stride로 조절) ---
    #     patch_hw, stride = 32, 16  # patch_hw: {24, 32, 48}, stride: {8, 16, 32}
    #     train_descs = [_extract_grid_sift(img, patch_hw, stride) for img in x_train]  # 학습용 BoF 입력
    #     test_descs = [_extract_grid_sift(img, patch_hw, stride) for img in x_test]  # 평가용 BoF 입력

    #     # (3) 모든 descriptor를 한데 모아 K-Means Codebook 생성 → BoF 벡터 생성
    #     # --- PCA 사용하지 않음 → PCA 후 모든 PCA축을 다 사용하는 구조면 연산비용만 증가하고 실익이 없을 것이라 판단이 들었음.
    #     flattened = np.vstack(train_descs)
    #     codebook = VisualCodebook(k=80)  # k=visual word 수 (20/40/80 등)
    #     codebook.fit(flattened)
    #     encoder = BoFEncoder(k=codebook.k, mode="hard", normalize="l1")

    #     hist_root = result_dir / "p1_hist"
    #     train_hist_dir = hist_root / "train"
    #     test_hist_dir = hist_root / "test"

    #     # --- 학습/테스트 이미지를 BoF 히스토그램으로 변환하고 PNG 저장 (등간격 30개 샘플) ---
    #     train_hist_indices = sorted(
    #         set(np.linspace(0, len(train_descs) - 1, num=min(30, len(train_descs)), dtype=int).tolist() if train_descs else [])
    #     )
    #     test_hist_indices = sorted(
    #         set(np.linspace(0, len(test_descs) - 1, num=min(30, len(test_descs)), dtype=int).tolist() if test_descs else [])
    #     )
    #     train_bow_list: List[np.ndarray] = []
    #     for idx, desc in enumerate(train_descs):
    #         vec = encoder.encode(codebook.transform(desc))
    #         train_bow_list.append(vec)
    #         if idx in train_hist_indices:
    #             save_histogram(
    #                 vec,
    #                 train_hist_dir / f"{idx:05d}_{x_train_names[idx]}.png",
    #                 f"Train {x_train_names[idx]}",
    #             )

    #     test_bow_list: List[np.ndarray] = []
    #     for idx, desc in enumerate(test_descs):
    #         vec = encoder.encode(codebook.transform(desc))
    #         test_bow_list.append(vec)
    #         if idx in test_hist_indices:
    #             save_histogram(
    #                 vec,
    #                 test_hist_dir / f"{idx:05d}_{x_test_names[idx]}.png",
    #                 f"Test {x_test_names[idx]}",
    #             )

    #     X_train_bow = np.vstack(train_bow_list)
    #     X_test_bow = np.vstack(test_bow_list)
    #     y_train_arr = np.asarray(y_train, dtype=np.int64)
    #     y_test_arr = np.asarray(y_test, dtype=np.int64)

    #     # ML 모델 Hyperopt Runner
    #     runner = HyperoptRunner()

    #     # Train/Valid 분할
    #     X_tr, X_val, y_tr, y_val = train_test_split(
    #         X_train_bow,
    #         y_train_arr,
    #         test_size=0.2,
    #         random_state=2025,
    #         stratify=y_train_arr,
    #     )

    #     # Baseline 모델 파라미터
    #     classical_models = {
    #         "svm": {"params": {"C": 1.0, "kernel": "linear"}},
    #         "rf": {"params": {"n_estimators": 500, "max_depth": None}},
    #         "xgb": {"params": {"n_estimators": 500, "learning_rate": 0.05}},
    #     }

    #     logger.info("---- [문제1] 파라미터 탐색 / 최종 스코어 / 시각화 ----")
    #     p1_records: List[Dict[str, Any]] = []
    #     for model_name, model_cfg in classical_models.items():
    #         save_dir = result_dir / f"p1_{model_name}"
    #         trainer_cfg = TrainerConfig(save_dir=save_dir)

    #         # Baseline 모델 학습 (체크포인트 존재 시 재학습 스킵)
    #         baseline_model = create_classical_model(model_name, params=model_cfg["params"])
    #         baseline_trainer = MLTrainer(model=baseline_model, config=trainer_cfg)
    #         baseline_ckpt = save_dir / f"{model_name}_baseline.pkl"
    #         # 저장된 checkpoint가 있으면 재학습 없이 로드
    #         if baseline_ckpt.exists():
    #             logger.info("[문제1][%s] Baseline checkpoint 로드: %s", model_name.upper(), baseline_ckpt)
    #             baseline_trainer.load_model(baseline_ckpt)
    #         else:
    #             baseline_trainer.fit(X_train_bow, y_train_arr)
    #             baseline_trainer.save(baseline_ckpt.name)

    #         # Baseline 모델 평가
    #         baseline_preds = baseline_trainer.predict(X_test_bow)
    #         baseline_stats = compute_classification_metrics(y_test_arr, baseline_preds, classes)
    #         # 평가결과 저장
    #         plot_confusion_matrix(baseline_stats["confusion_matrix"], classes, save_dir / f"confusion_baseline_{model_name}.png")
    #         logger.info(
    #             "[문제1][%s][Baseline] Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f\n%s",
    #             model_name.upper(),
    #             baseline_stats["accuracy"],
    #             baseline_stats["precision"],
    #             baseline_stats["recall"],
    #             baseline_stats["f1"],
    #             baseline_stats["confusion_matrix"],
    #         )
    #         p1_records.append(metrics_to_record(model_name.upper(), "baseline", baseline_stats))

    #         # Hyperopt 탐색 → 최적 모델로 전체 학습 후 평가
    #         tuned_trainer = MLTrainer(
    #             model=create_classical_model(model_name, params=model_cfg["params"]),
    #             config=trainer_cfg,
    #         )

    #         # 최적 모델 저장경로
    #         tuned_ckpt = save_dir / f"{model_name}_tuned.pkl"
    #         tuned_param_path = save_dir / f"{model_name}_tuned_params.json"
    #         logs_df = None
    #         best_params = None
    #         # Hyperopt 결과/모델이 이미 존재하면 탐색 단계를 건너뜀
    #         if tuned_ckpt.exists():
    #             tuned_trainer.load_model(tuned_ckpt)
    #             best_params = load_json(tuned_param_path)
    #             logger.info(
    #                 "[문제1][%s] 튜닝 모델 체크포인트 로드 완료 (Hyperopt 생략): %s",
    #                 model_name.upper(),
    #                 tuned_ckpt,
    #             )
    #         else:
    #             _, best_params, logs_df = tuned_trainer.hyperopt_search(
    #                 model_type=model_name,
    #                 search_runner=runner,
    #                 X_train=X_tr,
    #                 y_train=y_tr,
    #                 X_val=X_val,
    #                 y_val=y_val,
    #                 max_evals=100,
    #             )
    #             tuned_trainer.save(tuned_ckpt.name)
    #             if best_params:
    #                 save_json(best_params, tuned_param_path)

    #         if best_params:
    #             logger.info("[문제1][%s] Hyperopt Best Params=%s", model_name.upper(), best_params)
    #         elif tuned_ckpt.exists():
    #             logger.info("[문제1][%s] Hyperopt 파라미터 파일을 찾지 못했습니다.", model_name.upper())

    #         # 최적 모델 평가
    #         tuned_preds = tuned_trainer.predict(X_test_bow)
    #         tuned_stats = compute_classification_metrics(y_test_arr, tuned_preds, classes)
    #         # 평가결과 저장
    #         plot_confusion_matrix(tuned_stats["confusion_matrix"], classes, save_dir / f"confusion_tuned_{model_name}.png")
    #         logger.info(
    #             "[문제1][%s][Tuned] Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f\n%s",
    #             model_name.upper(),
    #             tuned_stats["accuracy"],
    #             tuned_stats["precision"],
    #             tuned_stats["recall"],
    #             tuned_stats["f1"],
    #             tuned_stats["confusion_matrix"],
    #         )
    #         p1_records.append(metrics_to_record(model_name.upper(), "tuned", tuned_stats))

    #         if logs_df is not None and not logs_df.empty:
    #             # --- Hyperopt 탐색 로그/시각화 저장 ---
    #             logs_df["model_type"] = model_name
    #             logs_df.to_csv(save_dir / f"hyperopt_logs_{model_name}.csv", index=False)

    #             # 산점도: 변수 vs F1 Score
    #             plot_param_performance_curves(
    #                 logs_df,
    #                 metric_col="f1_score",
    #                 save_dir=save_dir / "hyperopt_plots",
    #                 prefix=f"p1_{model_name}",
    #                 ignore_cols=["loss", "status", "model_type"],
    #             )
    #             # 파라미터 조합별 평균 히트맵
    #             plot_param_heatmaps(
    #                 logs_df,
    #                 metric_col="f1_score",
    #                 save_dir=save_dir / "hyperopt_plots",
    #                 prefix=f"p1_{model_name}",
    #                 ignore_cols=["loss", "status", "model_type"],
    #             )
    #             # Parallel Coordinates plot: 여러 파라미터를 동시에 확인
    #             plot_parallel_param_coordinates(
    #                 logs_df,
    #                 metric_col="f1_score",
    #                 save_path=save_dir / "hyperopt_plots" / f"p1_{model_name}_parallel.png",
    #                 ignore_cols=["loss", "status", "model_type"],
    #             )

    #             logger.info(
    #                 "[문제1][%s] F1 Score Range: %.4f ~ %.4f (mean=%.4f)",
    #                 model_name.upper(),
    #                 logs_df["f1_score"].min(),
    #                 logs_df["f1_score"].max(),
    #                 logs_df["f1_score"].mean(),
    #             )
    #     # --- 모든 실험 결과를 DataFrame으로 변환해 CSV 저장 ---
    #     if p1_records:
    #         p1_df = pd.DataFrame(p1_records)
    #         p1_df_path = result_dir / "p1_metrics.csv"
    #         p1_df.to_csv(p1_df_path, index=False)
    #         logger.info("[문제1] Metrics Summary:\n%s", p1_df.to_string(index=False))

    # else:
    #     logger.warning("[문제1] 데이터셋을 찾을 수 없습니다: %s", scene_dir)

    # ------------------------------------------------------------------
    # 2. 문제 2 — EuroSAT Classification (MLP/CNN/ResNet/ConvNeXt)
    #     • Parameter 결과: 각 모델 Hyperopt 로그/최적 파라미터
    #     • 최종 스코어: Baseline/Test Accuracy (Train/Test 모두 로깅)
    #     • 시각화: Train/Test Confusion Matrix 저장
    # ------------------------------------------------------------------
    logger.info("==================== 문제 2: EuroSAT Classification ====================")

    euro_cfg = EuroSATConfig(root=data_dir, img_size=32, augment=True)
    try:
        train_loader, train_eval_loader, val_loader, test_loader, labels = load_eurosat(euro_cfg)

    except Exception:  # pragma: no cover
        logger.error("[문제2] EuroSAT 로딩 실패", exc_info=True)
        train_loader = train_eval_loader = val_loader = test_loader = None
        labels = []

    if all(x is not None for x in [train_loader, train_eval_loader, val_loader, test_loader]):
        logger.info(
            "[문제2] TrainAug=%d | TrainEval=%d | Val=%d | Test=%d",
            len(train_loader.dataset),
            len(train_eval_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )

        # --- Train/Valid/Test 라벨 분포 ---
        _log_split_distribution(train_loader, "train_aug", len(labels))
        _log_split_distribution(train_eval_loader, "train_eval", len(labels))
        _log_split_distribution(val_loader, "val", len(labels))
        _log_split_distribution(test_loader, "test", len(labels))

        dl_runner = HyperoptRunner()
        dl_metrics_summary: Dict[str, Dict[str, Any]] = {}
        p2_records: List[Dict[str, Any]] = []
        label_list = list(labels)
        logger.info("---- [문제2] 파라미터 탐색 / 최종 스코어 / 시각화 ----")
        # for model_name in ["mlp", "cnn", "resnet", "convnext"]:
        for model_name in ["resnet", "convnext", "cnn", "mlp"]:
            save_dir = result_dir / f"p2_{model_name}"

            base_model = build_dl_model(model_name, len(labels))
            baseline_trainer = TorchTrainer(
                model=base_model,
                config=TrainerConfig(save_dir=save_dir),
                optimizer=torch.optim.Adam(base_model.parameters(), lr=1e-3),
                early_stopping_patience=10,
                model_name=f"{model_name}_baseline",
            )
            baseline_ckpt = save_dir / f"{model_name}_baseline.pt"
            if baseline_ckpt.exists():
                logger.info("[문제2][%s] Baseline checkpoint 로드: %s", model_name.upper(), baseline_ckpt)
                baseline_trainer.load_model(baseline_ckpt)
            else:
                baseline_trainer.fit(train_loader, val_loader, num_epochs=20)
                baseline_trainer.save(baseline_ckpt.name)

            # Baseline 모델 평가 지표
            base_train_stats = _evaluate_dl_split(baseline_trainer, train_eval_loader, label_list, "train_baseline", save_dir, model_name)
            base_test_stats = _evaluate_dl_split(baseline_trainer, test_loader, label_list, "test_baseline", save_dir, model_name)

            tuned_param_path = save_dir / f"{model_name}_tuned_params.json"
            best_params = load_json(tuned_param_path) if tuned_param_path.exists() else None
            tuned_model = build_dl_model(model_name, len(labels), best_params)
            tuned_trainer = TorchTrainer(
                model=tuned_model,
                config=TrainerConfig(save_dir=save_dir),
                optimizer=torch.optim.Adam(tuned_model.parameters(), lr=1e-3),
                early_stopping_patience=10,
                model_name=f"{model_name}_tuned",
            )
            tuned_ckpt = save_dir / f"{model_name}_tuned.pt"
            logs_df = None
            if tuned_ckpt.exists() and best_params is not None:
                tuned_trainer.load_model(tuned_ckpt)
                logger.info(
                    "[문제2][%s] 튜닝 모델 체크포인트 로드 완료 (Hyperopt 생략): %s",
                    model_name.upper(),
                    tuned_ckpt,
                )
            else:
                if tuned_ckpt.exists() and best_params is None:
                    logger.warning(
                        "[문제2][%s] Hyperopt 파라미터 파일이 없어 튜닝 체크포인트를 재생성합니다.",
                        model_name.upper(),
                    )
                _, best_params, logs_df = tuned_trainer.hyperopt_search(
                    model_type=model_name,
                    search_runner=dl_runner,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    max_evals=10,
                )
                tuned_trainer.save(tuned_ckpt.name)
                if best_params:
                    save_json(best_params, tuned_param_path)

            if best_params:
                logger.info("[문제2][%s] Hyperopt Best Params=%s", model_name.upper(), best_params)
            elif tuned_ckpt.exists():
                logger.info("[문제2][%s] Hyperopt 파라미터 파일을 찾지 못했습니다.", model_name.upper())

            # 최적 모델 평가 지표
            tuned_train_stats = _evaluate_dl_split(tuned_trainer, train_eval_loader, label_list, "train_tuned", save_dir, model_name)
            tuned_test_stats = _evaluate_dl_split(tuned_trainer, test_loader, label_list, "test_tuned", save_dir, model_name)

            if logs_df is not None and not logs_df.empty:
                logs_df["model_type"] = model_name
                logs_df.to_csv(save_dir / f"hyperopt_logs_{model_name}.csv", index=False)
                plot_param_performance_curves(
                    logs_df,
                    metric_col="f1_score",
                    save_dir=save_dir / "hyperopt_plots",
                    prefix=f"p2_{model_name}",
                    ignore_cols=["loss", "status", "model_type"],
                )
                plot_param_heatmaps(
                    logs_df,
                    metric_col="f1_score",
                    save_dir=save_dir / "hyperopt_plots",
                    prefix=f"p2_{model_name}",
                    ignore_cols=["loss", "status", "model_type"],
                )
                plot_parallel_param_coordinates(
                    logs_df,
                    metric_col="f1_score",
                    save_path=save_dir / "hyperopt_plots" / f"p2_{model_name}_parallel.png",
                    ignore_cols=["loss", "status", "model_type"],
                )

            dl_metrics_summary[model_name] = {"baseline": base_test_stats, "tuned": tuned_test_stats}
            p2_records.append(metrics_to_record(model_name.upper(), "baseline_test", base_test_stats))
            p2_records.append(metrics_to_record(model_name.upper(), "tuned_test", tuned_test_stats))

        if p2_records:
            p2_df = pd.DataFrame(p2_records)
            p2_df_path = result_dir / "p2_metrics.csv"
            p2_df.to_csv(p2_df_path, index=False)
            logger.info("[문제2] Test Metrics Summary:\n%s", p2_df.to_string(index=False))

        best_model_name = max(dl_metrics_summary, key=lambda k: dl_metrics_summary[k]["tuned"]["f1"])
        logger.info(
            "[문제2] 최고 성능 모델: %s (Baseline F1 %.4f / Tuned F1 %.4f)",
            best_model_name.upper(),
            dl_metrics_summary[best_model_name]["baseline"]["f1"],
            dl_metrics_summary[best_model_name]["tuned"]["f1"],
        )

    else:
        logger.warning("[문제2] EuroSAT 데이터 로더를 초기화하지 못했습니다.")

    end_time = datetime.now()
    logger.info("No. 1,2 of HW3 Elapsed Time: {} Minutes".format(round((end_time - st_time).seconds / 60, 2)))

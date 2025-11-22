"""
Detectron2 평가 모듈
    - COCOEvaluator + inference_on_dataset
"""

from __future__ import annotations

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model


def evaluate_model(cfg, dataset_name: str, weights_path: str | None = None):
    """COCOEvaluator로 모델을 평가한다.

    Args:
        cfg: Detectron2 config 객체.
        dataset_name (str): 평가할 DatasetCatalog 이름.
        weights_path (str | None): 사용할 가중치 경로. None이면 cfg.MODEL.WEIGHTS 사용.

    Returns:
        Dict[str, float]: AP/AR 등의 평가 결과.
    """
    cfg = cfg.clone()
    if weights_path is not None:
        cfg.MODEL.WEIGHTS = weights_path

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    evaluator = COCOEvaluator(
        dataset_name,
        distributed=False,
        output_dir=cfg.OUTPUT_DIR,
    )
    loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(model, loader, evaluator)

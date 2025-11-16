"""
Detectron2 Config 생성 모듈: Mask R-CNN 모델을 선택해 config 생성
"""

from __future__ import annotations

from typing import Optional

from detectron2 import model_zoo
from detectron2.config import get_cfg


def build_config(
    model_name: str,
    train_dataset: str,
    output_dir: str = "./detectron2_output",
    base_lr: float = 0.00025,
    max_iter: int = 1000,
    ims_per_batch: int = 2,
) -> "CfgNode":
    """
    Mask R-CNN 등 Detectron2 모델을 Fine-tuning하기 위한 config 생성.

    Args:
        model_name (str): model_zoo config yaml 이름 (예: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        train_dataset (str): DatasetCatalog 이름 (예: "balloon_train")
        output_dir (str): 학습 결과 저장 경로
        base_lr (float): Learning rate
        max_iter (int): Max iteration
        ims_per_batch (int): batch size

    Returns:
        CfgNode: Detectron2 config 객체
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))

    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # no LR decay

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # balloon only

    cfg.OUTPUT_DIR = output_dir

    return cfg

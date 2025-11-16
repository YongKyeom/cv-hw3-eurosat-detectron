"""
풍선(Balloon) 데이터셋 로더
Detectron2 COCO-style dictionary 포맷으로 변환하여 DatasetCatalog에 등록하기 위함.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from detectron2.structures import BoxMode


def load_balloon_dataset(img_dir: str | Path) -> List[Dict[str, Any]]:
    """
    balloon/train, balloon/val 디렉토리에서 JSON 파일을 로드하여 Detectron2 형식으로 변환한다.

    Args:
        img_dir (str | Path): balloon/train 또는 balloon/val 디렉토리 경로

    Returns:
        List[Dict[str, Any]]: Detectron2 dataset dicts
    """
    img_dir = Path(img_dir)
    json_path = img_dir / "via_region_data.json"

    with open(json_path) as f:
        annotations = json.load(f)

    dataset_dicts = []

    for idx, v in enumerate(annotations.values()):
        record = {}

        filename = img_dir / v["filename"]
        height, width = cv2.imread(str(filename)).shape[:2]

        record["file_name"] = str(filename)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        regions = v["regions"]
        for r in regions.values():
            shape = r["shape_attributes"]

            px = shape["all_points_x"]
            py = shape["all_points_y"]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly_flat = [p for xy in poly for p in xy]

            obj = {
                "bbox": [min(px), min(py), max(px), max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly_flat],
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

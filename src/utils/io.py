from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np


def load_color(path: str | Path) -> np.ndarray:
    """
    BGR 컬러 이미지를 로드한다.

    Args:
        path (str | Path): 이미지 파일 경로.

    Returns:
        np.ndarray: BGR 이미지 배열.

    Raises:
        FileNotFoundError: 이미지 로딩 실패 시.

    Note:
        - OpenCV는 기본적으로 BGR 채널 순서를 사용.
        - 존재하지 않는 파일이면 None을 반환하므로 예외 처리 필요.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    return img


def save_image(path: str | Path, img: np.ndarray) -> None:
    """
    이미지를 디스크에 저장한다. 상위 디렉토리를 자동 생성한다.

    Args:
        path (str | Path): 저장할 파일 경로.
        img (np.ndarray): 저장할 BGR 혹은 Gray 이미지.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    입력 이미지가 컬러일 경우 Gray로 변환한다.

    Args:
        img (np.ndarray): BGR 또는 Gray 이미지.

    Returns:
        np.ndarray: Gray 이미지.

    """
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_csv(
    path: str | Path,
    header: Sequence[str] | None,
    rows: Iterable[Sequence[object]],
) -> None:
    """
    CSV 파일 저장 함수.

    Args:
        path (str | Path): 저장 경로.
        header (Sequence[str] | None): 컬럼명. None이면 헤더 생략.
        rows (Iterable[Sequence[object]]): 데이터 행(iterable).

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def save_text_lines(path: str | Path, lines: List[str]) -> None:
    """
    텍스트 라인 목록을 파일로 저장한다.

    Args:
        path (str | Path): 저장 파일 경로.
        lines (List[str]): 저장할 문자열 목록.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def ensure_dir(path: Path) -> None:
    """필요한 디렉토리가 없으면 생성한다.

    Args:
        path (Path): 생성할 대상 디렉토리 경로.
    """
    path.mkdir(parents=True, exist_ok=True)


def _json_default(obj: Any) -> Any:
    """json.dump 에서 처리하지 못하는 객체를 문자열/리스트로 변환한다."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """dict 객체를 JSON 파일로 저장한다.

    Args:
        data (Dict[str, Any]): 저장할 데이터.
        path (str | Path): JSON 파일 경로.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Optional[Dict[str, Any]]:
    """JSON 파일을 불러와 dict 로 반환한다.

    Args:
        path (str | Path): 읽을 JSON 경로.

    Returns:
        Optional[Dict[str, Any]]: 파싱된 dict. 파일이 없으면 None.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_first_image(folder: str | Path, patterns: Sequence[str] | None = None) -> Optional[Path]:
    """지정한 폴더에서 가장 먼저 발견되는 이미지 파일을 반환한다.

    Args:
        folder (str | Path): 탐색할 상위 디렉토리.
        patterns (Sequence[str] | None): glob 패턴 목록. None이면 기본 확장자 사용.

    Returns:
        Optional[Path]: 발견된 이미지 경로. 없으면 None.
    """
    folder = Path(folder)
    if patterns is None:
        patterns = ("*.jpg", "*.jpeg", "*.png")

    for pattern in patterns:
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None

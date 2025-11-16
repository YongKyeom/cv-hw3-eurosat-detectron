from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Paths:
    """
    프로젝트 공통 경로 집합.

    Attributes:
        log_dir (Path): 실행 로그 저장 디렉토리 (root/log)
        data_dir (Path): 데이터 저장 디렉토리 (root/data)
        result_dir (Path): HW3 결과물이 저장되는 디렉토리 (root/result/hw3)
    """

    log_dir: Path
    data_dir: Path
    result_dir: Path

    @staticmethod
    def from_root(root: Union[str, Path]) -> "Paths":
        """
        루트 경로를 기준으로 경로 집합을 생성한다.
        필요한 디렉토리가 없으면 자동 생성한다.

        Args:
            root (Union[str, Path]): 프로젝트 루트 경로.

        Returns:
            Paths: log/data/result 디렉토리를 포함한 Paths 객체.

        Note:
            - HW1/HW2 코드 스타일을 그대로 계승했다.
            - HW3의 결과 폴더는 result/hw3 로 고정한다.
        """
        root = Path(root).resolve()

        # log, data, result 디렉토리 생성
        log_dir = root / "log"
        data_dir = root / "data"
        result_dir = root / "result" / "hw3"

        log_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        return Paths(
            log_dir=log_dir,
            data_dir=data_dir,
            result_dir=result_dir,
        )

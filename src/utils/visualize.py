from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    save_path: str | Path,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> None:
    """
    Confusion Matrix를 이미지로 저장한다.

    Args:
        cm (np.ndarray): (C, C) confusion matrix.
        labels (Sequence[str]): 클래스 이름 목록.
        save_path (str | Path): 저장 경로.
        figsize (Tuple[int, int]): 그림 크기.
        cmap (str): 색상맵.

    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_keypoints(
    image: np.ndarray,
    keypoints: Sequence[cv2.KeyPoint],
    save_path: str | Path,
) -> None:
    """
    이미지 위에 키포인트를 그려서 저장한다.

    Args:
        image (np.ndarray): BGR 이미지.
        keypoints (Sequence[cv2.KeyPoint]): 키포인트 리스트.
        save_path (str | Path): 저장 파일 경로.

    """
    vis = cv2.drawKeypoints(image, list(keypoints), None)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


def montage(
    images: Sequence[np.ndarray],
    cols: int = 8,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    이미지 목록을 타일 형태의 몽타주로 합친다.

    Args:
        images (Sequence[np.ndarray]): 합칠 이미지 리스트.
        cols (int): 한 줄에 놓을 이미지 수.
        gap (int): 이미지 간 간격(픽셀).
        bg_color (Tuple[int, int, int]): 배경색(BGR).

    Returns:
        np.ndarray: 몽타주 이미지.

    Note:
        - HW2 FeatureDescriptors.render_patches_montage 구현을 확장.
    """
    if len(images) == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)

    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols

    canvas_h = rows * h + (rows - 1) * gap
    canvas_w = cols * w + (cols - 1) * gap

    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = r * (h + gap)
        x = c * (w + gap)
        canvas[y : y + h, x : x + w] = img

    return canvas


def save_montage(
    images: Sequence[np.ndarray],
    save_path: str | Path,
    cols: int = 8,
    gap: int = 2,
) -> None:
    """
    이미지 몽타주를 생성하여 저장한다.

    Args:
        images (Sequence[np.ndarray]): 이미지 리스트.
        save_path (str | Path): 저장 경로.
        cols (int): 한 줄당 이미지 수.
        gap (int): 간격(픽셀).

    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = montage(images, cols=cols, gap=gap)
    cv2.imwrite(str(save_path), canvas)


def save_histogram(
    vec: np.ndarray,
    save_path: str | Path,
    title: str,
    figsize: Tuple[int, int] = (6, 3),
) -> None:
    """BoF/통계 히스토그램을 막대 그래프로 저장한다.

    Args:
        vec (np.ndarray): 시각화할 히스토그램 벡터.
        save_path (str | Path): 저장 경로.
        title (str): 그래프 제목(이미지 이름 등).
        figsize (Tuple[int, int]): 그래프 크기 설정.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(vec)), vec, color="steelblue")
    plt.title(title)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_param_performance_curves(
    logs_df: pd.DataFrame,
    metric_col: str,
    save_dir: str | Path,
    prefix: str,
    ignore_cols: Sequence[str] | None = None,
) -> None:
    """Hyperopt 로그에서 파라미터 대비 지표 변화를 산점도로 저장한다.

    Args:
        logs_df (pd.DataFrame): Hyperopt 탐색 로그 DataFrame.
        metric_col (str): y축에 사용할 지표 컬럼명(e.g., "f1_score").
        save_dir (str | Path): 그래프 저장 경로.
        prefix (str): 파일명/제목에 붙일 접두사.
        ignore_cols (Sequence[str] | None): 제외할 컬럼명 리스트.
    """
    if logs_df is None or logs_df.empty or metric_col not in logs_df:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ignored = set(ignore_cols or [])
    ignored.update({metric_col})
    numeric_cols = [
        col
        for col in logs_df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(logs_df[col])
    ]
    if not numeric_cols:
        return

    metric_series = logs_df[metric_col]
    for col in numeric_cols:
        plt.figure(figsize=(5, 3))
        sns.scatterplot(x=logs_df[col], y=metric_series)
        plt.title(f"{prefix} {col} vs {metric_col}")
        plt.xlabel(col)
        plt.ylabel(metric_col)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_{col}_vs_{metric_col}.png")
        plt.close()


def _discretize_series(series: pd.Series, bins: int) -> pd.Series:
    """연속형 파라미터를 히트맵 축에 쓰기 위해 구간화한다."""

    if series.empty:
        return pd.Series(pd.Categorical([]), index=series.index)

    non_null = series.dropna()
    unique_vals = np.sort(non_null.unique())
    if len(unique_vals) == 0:
        cat = pd.Categorical([np.nan] * len(series))
    elif len(unique_vals) <= bins:
        cat = pd.Categorical(series, categories=unique_vals, ordered=True)
    else:
        effective_bins = max(1, min(bins, len(unique_vals)))
        cat = pd.cut(series, bins=effective_bins, duplicates="drop")

    if isinstance(cat, pd.Series):
        result = cat.copy()
    else:
        result = pd.Series(cat, index=series.index)
    return result


def plot_param_heatmaps(
    logs_df: pd.DataFrame,
    metric_col: str,
    save_dir: str | Path,
    prefix: str,
    ignore_cols: Sequence[str] | None = None,
    bins: int = 8,
    max_plots: int = 6,
) -> None:
    """Hyperopt 로그에서 2개 파라미터 조합별 평균 지표 히트맵을 그린다."""

    if logs_df is None or logs_df.empty or metric_col not in logs_df:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ignored = set(ignore_cols or [])
    ignored.update({metric_col})
    numeric_cols = [
        col
        for col in logs_df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(logs_df[col])
    ]
    if len(numeric_cols) < 2:
        return

    combos = list(combinations(numeric_cols, 2))[:max_plots]
    for x_col, y_col in combos:
        data = logs_df[[x_col, y_col, metric_col]].dropna()
        if data.empty:
            continue
        x_bins = _discretize_series(data[x_col], bins)
        y_bins = _discretize_series(data[y_col], bins)
        x_categories = list(x_bins.cat.categories) if hasattr(x_bins, "cat") else []
        y_categories = list(y_bins.cat.categories) if hasattr(y_bins, "cat") else []
        if len(x_categories) == 0 or len(y_categories) == 0:
            continue

        data = data.assign(x_bin=x_bins, y_bin=y_bins)
        pivot = data.pivot_table(
            index="y_bin",
            columns="x_bin",
            values=metric_col,
            aggfunc="mean",
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako")
        plt.title(f"{prefix} {y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_heatmap_{y_col}_vs_{x_col}.png")
        plt.close()


def plot_parallel_param_coordinates(
    logs_df: pd.DataFrame,
    metric_col: str,
    save_path: str | Path,
    ignore_cols: Sequence[str] | None = None,
    sample_size: int = 200,
    num_bins: int = 4,
) -> None:
    """여러 파라미터와 지표를 동시에 확인하기 위한 Parallel Coordinates plot을 저장한다."""

    if logs_df is None or logs_df.empty or metric_col not in logs_df:
        return

    ignored = set(ignore_cols or [])
    ignored.update({metric_col})
    numeric_cols = [
        col
        for col in logs_df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(logs_df[col])
    ]
    if len(numeric_cols) < 2:
        return

    data = logs_df[numeric_cols + [metric_col]].dropna().copy()
    if data.empty:
        return

    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)

    bins = min(num_bins, data[metric_col].nunique())
    if bins < 2:
        return

    data["metric_bucket"] = pd.qcut(data[metric_col], q=bins, duplicates="drop")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(data[numeric_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=data.index)
    scaled_df["metric_bucket"] = data["metric_bucket"]

    plt.figure(figsize=(8, 5))
    parallel_coordinates(scaled_df, class_column="metric_bucket", colormap="viridis")
    plt.title(f"Parallel Coordinates ({metric_col})")
    plt.ylabel("Scaled Values")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

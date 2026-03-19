#!/usr/bin/env python3
"""
统计方法异常检测适配器 —— 支持多种无模型方法

方法列表：
1. Isolation Forest (IForest)
2. 3-Sigma (Z-Score)
3. IQR (四分位距)
4. MAD (中位绝对偏差)

这些方法不需要 GPU，可在任意环境中运行。
用于建立统计方法基线，与 TSFM 方法（Timer/Sundial）对比。

用法：
    python statistical_adapter.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv \
        --method iforest \
        --point-name FI6101.PV
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--method", default="iforest",
                        choices=["iforest", "3sigma", "iqr", "mad"])
    parser.add_argument("--contamination", type=float, default=0.01,
                        help="IForest 预期异常比例")
    parser.add_argument("--threshold-k", type=float, default=3.0,
                        help="3sigma/MAD 阈值系数")
    parser.add_argument("--min-cluster", type=int, default=5,
                        help="最小连续异常点数（合并小段）")
    parser.add_argument("--n-downsample", type=int, default=50000,
                        help="降采样点数（0=不降采样）")
    return parser.parse_args()


def downsample_m4(values: np.ndarray, n_out: int) -> tuple:
    """M4 降采样：每段保留 min, max, first, last"""
    n = len(values)
    if n <= n_out:
        return values, np.arange(n)

    chunk_size = n // (n_out // 4)
    indices = []
    for i in range(0, n, chunk_size):
        chunk = values[i:i + chunk_size]
        if len(chunk) == 0:
            continue
        idx_base = i
        indices.extend([
            idx_base,  # first
            idx_base + np.argmin(chunk),  # min
            idx_base + np.argmax(chunk),  # max
            idx_base + len(chunk) - 1,  # last
        ])
    indices = sorted(set(indices))
    return values[indices], np.array(indices)


def detect_iforest(values: np.ndarray, contamination: float = 0.01) -> np.ndarray:
    """Isolation Forest 异常检测"""
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    pred = clf.fit_predict(values.reshape(-1, 1))
    return (pred == -1).astype(int)


def detect_3sigma(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """3-Sigma 异常检测"""
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-10:
        return np.zeros(len(values), dtype=int)
    z = np.abs(values - mean) / std
    return (z > k).astype(int)


def detect_iqr(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """IQR 异常检测"""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    if iqr < 1e-10:
        return np.zeros(len(values), dtype=int)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return ((values < lower) | (values > upper)).astype(int)


def detect_mad(values: np.ndarray, k: float = 3.5) -> np.ndarray:
    """MAD (Median Absolute Deviation) 异常检测"""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-10:
        mad = np.std(values)
    if mad < 1e-10:
        return np.zeros(len(values), dtype=int)
    z = np.abs(values - median) / (mad * 1.4826)
    return (z > k).astype(int)


def merge_short_clusters(mask: np.ndarray, min_cluster: int) -> np.ndarray:
    """移除过短的异常簇"""
    if min_cluster <= 1:
        return mask
    result = mask.copy()
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < min_cluster:
            result[s:e] = 0
    return result


def map_mask_to_original(mask_ds: np.ndarray, indices: np.ndarray,
                          original_length: int) -> np.ndarray:
    """将降采样后的 mask 映射回原始长度"""
    mask_orig = np.zeros(original_length, dtype=int)
    # 在原始索引位置标记
    anomaly_indices = indices[mask_ds == 1]
    # 对连续异常区间进行插值填充
    if len(anomaly_indices) == 0:
        return mask_orig

    # 简单策略：标记异常点及其相邻区间
    for idx in anomaly_indices:
        mask_orig[idx] = 1

    # 填充相邻异常点之间的间隙（如果间隙 < 原始数据的 chunk_size）
    chunk_size = max(1, original_length // len(indices))
    diff = np.diff(np.concatenate(([0], mask_orig, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for i in range(len(starts) - 1):
        gap = starts[i + 1] - ends[i]
        if gap <= chunk_size * 2:
            mask_orig[ends[i]:starts[i + 1]] = 1

    return mask_orig


def main():
    args = parse_args()

    try:
        # 加载数据
        df = pd.read_csv(args.input)
        skip = {"Time", "time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        value_col = [c for c in df.columns if c not in skip][0]
        values = df[value_col].values.astype(np.float64)

        # 处理 NaN
        nan_mask = np.isnan(values)
        if nan_mask.any():
            values[nan_mask] = np.nanmedian(values)

        original_length = len(values)
        print(f"[Statistical] 点位: {args.point_name}, 数据量: {original_length}, 方法: {args.method}")

        # 降采样
        if args.n_downsample > 0 and original_length > args.n_downsample:
            ds_values, ds_indices = downsample_m4(values, args.n_downsample)
            print(f"[Statistical] 降采样: {original_length} -> {len(ds_values)} 点")
        else:
            ds_values = values
            ds_indices = np.arange(original_length)

        # 检测
        st = time.time()
        method_map = {
            "iforest": lambda v: detect_iforest(v, args.contamination),
            "3sigma": lambda v: detect_3sigma(v, args.threshold_k),
            "iqr": lambda v: detect_iqr(v),
            "mad": lambda v: detect_mad(v, args.threshold_k),
        }
        mask_ds = method_map[args.method](ds_values)

        # 合并短簇
        mask_ds = merge_short_clusters(mask_ds, args.min_cluster)

        # 映射回原始长度
        if len(ds_values) < original_length:
            mask = map_mask_to_original(mask_ds, ds_indices, original_length)
        else:
            mask = mask_ds

        elapsed = time.time() - st
        anomaly_rate = float(mask.mean())
        print(f"[Statistical] 完成: rate={anomaly_rate:.4f}, {elapsed:.2f}s")

        # 保存
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        result = pd.DataFrame({"global_mask": mask.astype(np.int8)})
        result.to_csv(args.output, index=False)

        # 状态文件
        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({
                "status": "success",
                "anomaly_rate": anomaly_rate,
                "anomaly_count": int(mask.sum()),
                "total_rows": original_length,
                "elapsed": elapsed,
                "method": args.method,
            }, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        status_path = args.output.replace(".csv", ".status.json")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(status_path, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

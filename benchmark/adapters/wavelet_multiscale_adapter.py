#!/usr/bin/env python3
"""
小波多尺度异常检测

原理（参考 Meta-MWDG 2024）：
1. 离散小波分解（DWT）将时序分为多个频率分量
2. 高频分量（detail coefficients）→ 检测尖峰/跳变（用 MAD）
3. 低频分量（approximation）→ 检测趋势漂移（用滑动均值偏差）
4. 多尺度融合：各层异常分数加权合并

优势：
- 不需要 GPU，纯 CPU 计算
- 自然实现多尺度检测（快速异常 + 慢速异常）
- 与 Timer 互补（Timer 固定 256 窗口，小波自适应多尺度）
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def dwt_decompose(values: np.ndarray, wavelet: str = 'db4',
                   levels: int = 4) -> tuple:
    """离散小波分解"""
    import pywt
    coeffs = pywt.wavedec(values, wavelet, level=levels)
    # coeffs[0] = approximation (lowest freq)
    # coeffs[1:] = details (high to low freq)
    return coeffs


def detect_high_freq_anomalies(detail_coeffs: np.ndarray,
                                 k: float = 3.0) -> np.ndarray:
    """在高频分量中检测尖峰/跳变（MAD 方法）"""
    median = np.median(detail_coeffs)
    mad = np.median(np.abs(detail_coeffs - median))
    if mad < 1e-10:
        mad = np.std(detail_coeffs)
    if mad < 1e-10:
        return np.zeros(len(detail_coeffs))
    scores = np.abs(detail_coeffs - median) / (mad * 1.4826)
    return scores


def detect_low_freq_anomalies(approx_coeffs: np.ndarray,
                                window: int = 50) -> np.ndarray:
    """在低频分量中检测趋势漂移（滑动均值偏差）"""
    n = len(approx_coeffs)
    if n < window * 2:
        return np.zeros(n)

    # 滑动均值
    kernel = np.ones(window) / window
    smoothed = np.convolve(approx_coeffs, kernel, mode='same')
    deviation = np.abs(approx_coeffs - smoothed)

    # MAD 标准化
    median_dev = np.median(deviation)
    mad_dev = np.median(np.abs(deviation - median_dev))
    if mad_dev < 1e-10:
        mad_dev = np.std(deviation)
    if mad_dev < 1e-10:
        return np.zeros(n)

    scores = (deviation - median_dev) / (mad_dev * 1.4826)
    return np.maximum(scores, 0)


def multiscale_detect(values: np.ndarray, wavelet: str = 'db4',
                       levels: int = 4, high_k: float = 3.0,
                       low_window: int = 50,
                       threshold_percentile: float = 90) -> tuple:
    """多尺度异常检测"""
    import pywt

    # 分解
    coeffs = pywt.wavedec(values, wavelet, level=levels)

    # 各层异常分数
    all_scores = np.zeros(len(values))
    weights = []

    # 高频分量（从高到低频）
    for i, detail in enumerate(coeffs[1:]):
        scores = detect_high_freq_anomalies(detail, high_k)
        # 上采样回原始长度
        upsampled = np.repeat(scores, 2 ** (levels - i))[:len(values)]
        if len(upsampled) < len(values):
            upsampled = np.pad(upsampled, (0, len(values) - len(upsampled)))
        # 高频权重更高（尖峰更重要）
        weight = 1.0 / (i + 1)
        all_scores += upsampled * weight
        weights.append(weight)

    # 低频分量
    approx = coeffs[0]
    low_scores = detect_low_freq_anomalies(approx, low_window)
    upsampled_low = np.repeat(low_scores, 2 ** levels)[:len(values)]
    if len(upsampled_low) < len(values):
        upsampled_low = np.pad(upsampled_low, (0, len(values) - len(upsampled_low)))
    all_scores += upsampled_low * 0.5  # 低频权重

    # 归一化
    total_weight = sum(weights) + 0.5
    all_scores /= total_weight

    # 阈值
    threshold = np.percentile(all_scores[all_scores > 0], threshold_percentile) if (all_scores > 0).any() else 0
    mask = (all_scores > threshold).astype(int)

    return mask, all_scores


def downsample_m4(values, n_out):
    n = len(values)
    if n <= n_out:
        return values, np.arange(n)
    chunk_size = max(1, n // (n_out // 4))
    indices = []
    for i in range(0, n, chunk_size):
        chunk = values[i:i + chunk_size]
        if len(chunk) == 0:
            continue
        idx_base = i
        indices.extend([idx_base, idx_base + int(np.argmin(chunk)),
                        idx_base + int(np.argmax(chunk)), idx_base + len(chunk) - 1])
    indices = sorted(set(indices))
    return values[indices], np.array(indices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--n-downsample", type=int, default=10000)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--high-k", type=float, default=3.0)
    parser.add_argument("--threshold-percentile", type=float, default=90)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        df = pd.read_csv(args.input)
        skip = {"Time", "time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        if "Time" not in df.columns and "time" in df.columns:
            df.rename(columns={"time": "Time"}, inplace=True)
        value_col = [c for c in df.columns if c not in skip][0]
        values = df[value_col].values.astype(np.float64)
        nan_mask = np.isnan(values)
        if nan_mask.any():
            values[nan_mask] = np.nanmedian(values)

        original_length = len(values)

        if original_length > args.n_downsample:
            ds_values, ds_indices = downsample_m4(values, args.n_downsample)
        else:
            ds_values, ds_indices = values, np.arange(original_length)

        st = time.time()
        mask_ds, scores_ds = multiscale_detect(
            ds_values, args.wavelet, args.levels,
            args.high_k, threshold_percentile=args.threshold_percentile)
        elapsed = time.time() - st

        # 映射回原始长度
        if len(ds_values) < original_length:
            mask = np.zeros(original_length, dtype=int)
            for i, idx in enumerate(ds_indices):
                if mask_ds[i] == 1:
                    mask[idx] = 1
        else:
            mask = mask_ds

        rate = float(mask.mean())

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.compact:
            pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(args.output, index=False)
        else:
            pd.DataFrame({"Time": df["Time"], "value": df[value_col],
                          "global_mask": mask.astype(int)}).to_csv(args.output, index=False)

        with open(args.output.replace(".csv", ".status.json"), "w") as f:
            json.dump({"status": "success", "anomaly_rate": rate,
                       "anomaly_count": int(mask.sum()), "total_rows": original_length,
                       "elapsed": elapsed}, f, indent=2)

        print(f"[Wavelet] {args.point_name}: rate={rate:.4f}, {elapsed:.2f}s")

    except Exception as e:
        import traceback
        traceback.print_exc()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output.replace(".csv", ".status.json"), "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

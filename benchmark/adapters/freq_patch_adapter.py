#!/usr/bin/env python3
"""
频率域 Patching 异常检测（CATCH 思想的轻量实现）

核心思想（参考 CATCH ICLR 2025）：
1. FFT 将时序转为频率域
2. 将频谱分为多个 band（Frequency Patching）
3. 每个 band 独立检测异常（异常在特定频段有独特信号）
4. 多 band 异常分数融合

与完整 CATCH 的区别：
- 单变量（无 Channel Fusion）
- 无需训练（基于统计阈值，不需要 GPU）
- 适合快速评估频率域方法的潜力

优势：比时域 MAD 更好地捕获频率特征的异常
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def frequency_patch_detect(values: np.ndarray, num_patches: int = 8,
                            overlap: float = 0.5, window_size: int = 256,
                            step: int = 64, k: float = 2.5) -> tuple:
    """
    频率域 Patching 异常检测

    流程：
    1. 滑动窗口提取时序片段
    2. FFT 得到频谱
    3. 将频谱分为 num_patches 个 band
    4. 每个 band 用 MAD 检测异常
    5. 融合所有 band 的异常分数
    """
    n = len(values)
    if n < window_size:
        return np.zeros(n, dtype=int), np.zeros(n)

    # 标准化
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val < 1e-10:
        return np.zeros(n, dtype=int), np.zeros(n)
    normalized = (values - mean_val) / std_val

    # 滑动窗口 + FFT
    window_scores = []
    window_starts = []

    for start in range(0, n - window_size + 1, step):
        window = normalized[start:start + window_size]

        # FFT
        fft_vals = np.fft.rfft(window)
        fft_magnitude = np.abs(fft_vals)

        # 频率 Patching：将频谱分为 num_patches 个 band
        freq_len = len(fft_magnitude)
        patch_size = max(1, freq_len // num_patches)

        band_features = []
        for p in range(num_patches):
            p_start = p * patch_size
            p_end = min((p + 1) * patch_size, freq_len)
            band = fft_magnitude[p_start:p_end]
            if len(band) > 0:
                band_features.append({
                    'energy': float(np.sum(band ** 2)),
                    'peak': float(np.max(band)),
                    'mean': float(np.mean(band)),
                    'std': float(np.std(band)),
                })
            else:
                band_features.append({'energy': 0, 'peak': 0, 'mean': 0, 'std': 0})

        window_scores.append(band_features)
        window_starts.append(start)

    if not window_scores:
        return np.zeros(n, dtype=int), np.zeros(n)

    # 对每个 band 的每个特征做 MAD 异常检测
    num_windows = len(window_scores)
    anomaly_scores = np.zeros(num_windows)

    for p in range(num_patches):
        for feature_name in ['energy', 'peak', 'std']:
            feature_vals = np.array([ws[p][feature_name] for ws in window_scores])

            if np.std(feature_vals) < 1e-10:
                continue

            # MAD
            median = np.median(feature_vals)
            mad = np.median(np.abs(feature_vals - median))
            if mad < 1e-10:
                mad = np.std(feature_vals)
            if mad < 1e-10:
                continue

            z = np.abs(feature_vals - median) / (mad * 1.4826)
            anomaly_scores += np.maximum(z - k, 0)

    # 映射回原始时间步
    time_scores = np.zeros(n)
    time_count = np.zeros(n)
    for i, start in enumerate(window_starts):
        time_scores[start:start + window_size] += anomaly_scores[i]
        time_count[start:start + window_size] += 1

    time_count[time_count == 0] = 1
    time_scores /= time_count

    # 阈值
    positive = time_scores[time_scores > 0]
    if len(positive) == 0:
        return np.zeros(n, dtype=int), time_scores

    threshold = np.percentile(positive, 85)
    mask = (time_scores > threshold).astype(int)

    return mask, time_scores


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--n-downsample", type=int, default=10000)
    parser.add_argument("--num-patches", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        skip = {"Time", "time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        if "Time" not in df.columns and "time" in df.columns:
            df.rename(columns={"time": "Time"}, inplace=True)
        value_col = [c for c in df.columns if c not in skip][0]
        values = df[value_col].values.astype(np.float64)
        values[np.isnan(values)] = np.nanmedian(values)
        original_length = len(values)

        if original_length > args.n_downsample:
            ds_values, ds_indices = downsample_m4(values, args.n_downsample)
        else:
            ds_values, ds_indices = values, np.arange(original_length)

        st = time.time()
        mask_ds, scores_ds = frequency_patch_detect(
            ds_values, args.num_patches, window_size=args.window_size)
        elapsed = time.time() - st

        if len(ds_values) < original_length:
            mask = np.zeros(original_length, dtype=int)
            for i, idx in enumerate(ds_indices):
                if mask_ds[i] == 1:
                    mask[idx] = 1
        else:
            mask = mask_ds

        rate = float(mask.mean())

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(args.output, index=False)
        with open(args.output.replace(".csv", ".status.json"), "w") as f:
            json.dump({"status": "success", "anomaly_rate": rate,
                       "anomaly_count": int(mask.sum()), "total_rows": original_length,
                       "elapsed": elapsed}, f, indent=2)
        print(f"[FreqPatch] {args.point_name}: rate={rate:.4f}, {elapsed:.2f}s")

    except Exception as e:
        import traceback; traceback.print_exc()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output.replace(".csv", ".status.json"), "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

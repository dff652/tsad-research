#!/usr/bin/env python3
"""
MOMENT 适配器 —— 在 moment conda 环境中运行

MOMENT (CMU, ICML'24) 是 T5-based 时序基础模型，
通过重建范式进行零样本异常检测：
  异常分数 = |重建值 - 原始值|

用法：
    PYTHONNOUSERSITE=1 conda run -n moment python moment_adapter.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv \
        --point-name FI6101.PV
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--model-path", default="AutonLab/MOMENT-1-large")
    parser.add_argument("--seq-len", type=int, default=512, help="MOMENT 固定输入长度")
    parser.add_argument("--n-downsample", type=int, default=10000)
    parser.add_argument("--threshold-percentile", type=float, default=95,
                        help="异常分数的百分位数阈值")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def downsample_m4(values, n_out):
    """M4 降采样"""
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
        indices.extend([idx_base, idx_base + np.argmin(chunk),
                        idx_base + np.argmax(chunk), idx_base + len(chunk) - 1])
    indices = sorted(set(indices))
    return values[indices], np.array(indices)


def moment_anomaly_detect(values: np.ndarray, model, seq_len: int = 512,
                           threshold_percentile: float = 95) -> tuple:
    """使用 MOMENT 进行异常检测

    策略：将长时序切分为 seq_len 长度的窗口，逐窗口重建，
    计算残差作为异常分数，用百分位数阈值判定异常。
    """
    import torch

    n = len(values)
    anomaly_scores = np.zeros(n)
    count = np.zeros(n)  # 用于重叠区域取平均

    # 标准化
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val < 1e-8:
        std_val = 1.0
    normalized = (values - mean_val) / std_val

    # 滑动窗口（步长 = seq_len // 2，50% 重叠）
    step = seq_len // 2
    device = next(model.parameters()).device

    with torch.no_grad():
        for start in range(0, max(1, n - seq_len + 1), step):
            end = min(start + seq_len, n)
            window = normalized[start:end]

            # 补齐到 seq_len
            if len(window) < seq_len:
                padded = np.zeros(seq_len, dtype=np.float32)
                padded[:len(window)] = window
                actual_len = len(window)
            else:
                padded = window.astype(np.float32)
                actual_len = seq_len

            x = torch.tensor(padded).reshape(1, 1, seq_len).to(device)
            output = model(x_enc=x)
            recon = output.reconstruction.squeeze().cpu().numpy()

            # 计算残差
            residuals = np.abs(padded[:actual_len] - recon[:actual_len])
            anomaly_scores[start:start + actual_len] += residuals
            count[start:start + actual_len] += 1

    # 取平均（重叠区域）
    count[count == 0] = 1
    anomaly_scores /= count

    # 阈值判定
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    mask = (anomaly_scores > threshold).astype(int)

    return mask, anomaly_scores, threshold


def main():
    args = parse_args()

    try:
        # 加载数据
        df = pd.read_csv(args.input)
        skip = {"Time", "time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        if "Time" not in df.columns and "time" in df.columns:
            df.rename(columns={"time": "Time"}, inplace=True)
        value_col = [c for c in df.columns if c not in skip][0]
        values = df[value_col].values.astype(np.float64)
        original_length = len(values)

        print(f"[MOMENT] 点位: {args.point_name}, 数据量: {original_length}")

        # 降采样
        if original_length > args.n_downsample:
            ds_values, ds_indices = downsample_m4(values, args.n_downsample)
            print(f"[MOMENT] 降采样: {original_length} -> {len(ds_values)}")
        else:
            ds_values = values
            ds_indices = np.arange(original_length)

        # 加载模型
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained(
            args.model_path,
            model_kwargs={"task_name": "reconstruction"},
        )
        model.init()
        print("[MOMENT] 模型加载完成")

        # 检测
        st = time.time()
        mask_ds, scores_ds, threshold = moment_anomaly_detect(
            ds_values, model, seq_len=args.seq_len,
            threshold_percentile=args.threshold_percentile)
        elapsed = time.time() - st

        # 映射回原始长度
        if len(ds_values) < original_length:
            mask = np.zeros(original_length, dtype=int)
            for i, idx in enumerate(ds_indices):
                if mask_ds[i] == 1:
                    mask[idx] = 1
            # 填充相邻标记点之间的间隙
            anomaly_indices = ds_indices[mask_ds == 1]
            if len(anomaly_indices) > 1:
                chunk_size = max(1, original_length // len(ds_indices))
                diff = np.diff(np.concatenate(([0], mask, [0])))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                for si in range(len(starts) - 1):
                    if starts[si + 1] - ends[si] <= chunk_size * 2:
                        mask[ends[si]:starts[si + 1]] = 1
        else:
            mask = mask_ds

        anomaly_rate = float(mask.mean())
        print(f"[MOMENT] 完成: rate={anomaly_rate:.4f}, threshold={threshold:.4f}, {elapsed:.2f}s")

        # 保存
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.compact:
            pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(args.output, index=False)
        else:
            result = pd.DataFrame({
                "Time": df["Time"] if "Time" in df.columns else range(len(df)),
                "value": df[value_col],
                "global_mask": mask.astype(int),
            })
            result.to_csv(args.output, index=False)

        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({
                "status": "success",
                "anomaly_rate": anomaly_rate,
                "anomaly_count": int(mask.sum()),
                "total_rows": original_length,
                "elapsed": elapsed,
                "threshold": float(threshold),
            }, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

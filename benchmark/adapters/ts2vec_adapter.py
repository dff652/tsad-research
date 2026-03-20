#!/usr/bin/env python3
"""
TS2Vec 对比学习异常检测适配器

原理（AAAI 2022）：
1. 层级化对比学习预训练：学习时序的通用表示
2. 异常检测：对每个时间步，用有/无 mask 两次编码，表示距离 = 异常分数
3. 不依赖标注，纯自监督

训练策略：
- 用 337 训练集点位的数据预训练通用 TS2Vec 模型
- 测试集 99 点位直接推理

TS2Vec 支持 CPU（慢但可行）
"""

import os
import sys
import json
import glob
import re
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 添加 TS2Vec 目录
TS2VEC_DIR = str(Path(__file__).resolve().parent.parent.parent / "ts2vec")
sys.path.insert(0, TS2VEC_DIR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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


def load_train_data(data_dir: str, points_file: str, n_downsample: int = 2000,
                     max_points: int = 100) -> np.ndarray:
    """加载训练数据：多个点位的时序片段"""
    with open(points_file) as f:
        points = [l.strip() for l in f if l.strip()][:max_points]

    all_series = []
    for pt in points:
        matches = glob.glob(os.path.join(data_dir, f"global_adtk_hbos_*_{pt}_*_trend_seasonal_resid.csv"))
        if not matches:
            continue
        df = pd.read_csv(matches[0])
        skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        vcol = [c for c in df.columns if c not in skip][0]
        v = df[vcol].values.astype(np.float64)
        v[np.isnan(v)] = np.nanmedian(v) if not np.all(np.isnan(v)) else 0

        if len(v) > n_downsample:
            v, _ = downsample_m4(v, n_downsample)

        # 标准化
        mean, std = v.mean(), v.std()
        if std > 1e-10:
            v = (v - mean) / std
        all_series.append(v)

    # 统一长度（取最短或截断）
    if not all_series:
        return np.zeros((1, 1000, 1))
    min_len = min(len(s) for s in all_series)
    min_len = min(min_len, 2000)  # 限制最大长度
    data = np.array([s[:min_len] for s in all_series])
    return data.reshape(len(data), min_len, 1)  # (N, T, 1)


def ts2vec_anomaly_detect(model, values: np.ndarray,
                           threshold_percentile: float = 95) -> tuple:
    """用 TS2Vec 进行异常检测"""
    # 标准化
    mean, std = values.mean(), values.std()
    if std < 1e-10:
        return np.zeros(len(values), dtype=int), np.zeros(len(values))
    normalized = (values - mean) / std

    data = normalized.reshape(1, -1, 1)  # (1, T, 1)

    # 两次编码：有 mask 和无 mask
    repr_full = model.encode(data, mask='all_true')     # (1, T, D)
    repr_masked = model.encode(data, mask='mask_last')   # (1, T, D)

    # 异常分数 = 表示距离
    scores = np.linalg.norm(repr_full[0] - repr_masked[0], axis=1)  # (T,)

    # 阈值
    threshold = np.percentile(scores, threshold_percentile)
    mask = (scores > threshold).astype(int)

    return mask, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["train", "detect", "full"])
    parser.add_argument("--n-downsample", type=int, default=2000)
    parser.add_argument("--max-train-points", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "results/ts2vec_model"))
    args = parser.parse_args()

    data_dir = str(PROJECT_ROOT / "data/adtk_hbos_old")
    train_points = str(PROJECT_ROOT / "data/cleaned/train_points.txt")
    test_points_file = str(PROJECT_ROOT / "data/cleaned/evaluated_points.txt")
    output_dir = str(PROJECT_ROOT / "results/predictions/ts2vec")
    os.makedirs(output_dir, exist_ok=True)

    from ts2vec import TS2Vec

    if args.mode in ("train", "full"):
        print("[TS2Vec] 加载训练数据...")
        train_data = load_train_data(data_dir, train_points,
                                      args.n_downsample, args.max_train_points)
        print(f"[TS2Vec] 训练数据: {train_data.shape}")

        model = TS2Vec(
            input_dims=1, output_dims=320, hidden_dims=64, depth=10,
            device=args.device, lr=0.001, batch_size=8,
            max_train_length=1000,
        )

        st = time.time()
        model.fit(train_data, n_epochs=args.epochs, verbose=True)
        train_time = time.time() - st
        print(f"[TS2Vec] 训练完成: {train_time:.1f}s")

        model.save(args.model_path)
        print(f"[TS2Vec] 模型保存: {args.model_path}")
    else:
        model = TS2Vec(input_dims=1, output_dims=320, hidden_dims=64, depth=10,
                        device=args.device)
        model.load(args.model_path)

    if args.mode in ("detect", "full"):
        with open(test_points_file) as f:
            test_points = [l.strip() for l in f if l.strip()]

        print(f"[TS2Vec] 检测 {len(test_points)} 个测试点位...")
        success = errors = 0

        for i, pt in enumerate(test_points):
            matches = glob.glob(os.path.join(data_dir,
                        f"global_adtk_hbos_*_{pt}_*_trend_seasonal_resid.csv"))
            if not matches:
                errors += 1
                continue

            try:
                df = pd.read_csv(matches[0])
                skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                        "global_mask_cluster", "local_mask_cluster"}
                vcol = [c for c in df.columns if c not in skip][0]
                v = df[vcol].values.astype(np.float64)
                v[np.isnan(v)] = np.nanmedian(v) if not np.all(np.isnan(v)) else 0
                orig_len = len(v)

                if orig_len > args.n_downsample:
                    ds_v, ds_idx = downsample_m4(v, args.n_downsample)
                else:
                    ds_v, ds_idx = v, np.arange(orig_len)

                mask_ds, scores_ds = ts2vec_anomaly_detect(model, ds_v)

                if len(ds_v) < orig_len:
                    mask = np.zeros(orig_len, dtype=int)
                    for j, idx in enumerate(ds_idx):
                        if mask_ds[j] == 1:
                            mask[idx] = 1
                else:
                    mask = mask_ds

                rate = float(mask.mean())
                pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(
                    os.path.join(output_dir, f"{pt}.csv"), index=False)
                with open(os.path.join(output_dir, f"{pt}.status.json"), "w") as f:
                    json.dump({"status": "success", "anomaly_rate": rate,
                               "anomaly_count": int(mask.sum()),
                               "total_rows": orig_len}, f, indent=2)
                success += 1

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(test_points)}] success={success}")

            except Exception as e:
                errors += 1
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}] error: {e}")

        print(f"\n[TS2Vec] 完成: {success}/{len(test_points)}")
        files = glob.glob(os.path.join(output_dir, "*.status.json"))
        rates = []
        for f in files:
            d = json.load(open(f))
            if d.get("status") == "success":
                rates.append(d["anomaly_rate"])
        if rates:
            print(f"  mean_rate={np.mean(rates):.4f}, median={np.median(rates):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Teacher-Student 伪标签异常检测

流程：
1. Teacher = Timer 推理结果（已有 99 测试点位的 mask）
2. 用 Timer 的 mask 作为伪标签训练 Student（轻量模型）
3. Student = 改进的统计方法：在 Timer 标记的异常区间内学习特征阈值
4. Student 推理：用学到的阈值检测所有点位

关键创新：Student 不是简单复制 Teacher，而是学习 Teacher 的"判断标准"
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

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


def learn_from_teacher(data_dir: str, teacher_dir: str, points: list,
                        n_downsample: int = 5000) -> dict:
    """从 Teacher（Timer）的伪标签中学习异常特征阈值"""

    # 收集 Teacher 标记的异常区间和正常区间的统计特征
    anomaly_features = []  # 异常窗口的特征
    normal_features = []   # 正常窗口的特征
    window_size = 64

    for pt in points:
        # 加载 Teacher 的 mask
        teacher_file = os.path.join(teacher_dir, f"{pt}.csv")
        if not os.path.exists(teacher_file):
            continue

        matches = glob.glob(os.path.join(data_dir,
                    f"global_adtk_hbos_*_{pt}_*_trend_seasonal_resid.csv"))
        if not matches:
            continue

        df = pd.read_csv(matches[0])
        skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        vcol = [c for c in df.columns if c not in skip][0]
        values = df[vcol].values.astype(np.float64)
        values[np.isnan(values)] = np.nanmedian(values) if not np.all(np.isnan(values)) else 0

        teacher_mask = pd.read_csv(teacher_file)["global_mask"].values

        if len(values) > n_downsample:
            ds_v, ds_idx = downsample_m4(values, n_downsample)
            ds_mask = np.zeros(len(ds_v), dtype=int)
            for i, idx in enumerate(ds_idx):
                if idx < len(teacher_mask):
                    ds_mask[i] = teacher_mask[idx]
        else:
            ds_v = values
            ds_mask = teacher_mask[:len(values)]

        # 标准化
        mean_v, std_v = ds_v.mean(), ds_v.std()
        if std_v < 1e-10:
            continue
        norm_v = (ds_v - mean_v) / std_v

        # 提取窗口特征
        for start in range(0, len(norm_v) - window_size, window_size // 2):
            window = norm_v[start:start + window_size]
            mask_window = ds_mask[start:start + window_size]

            features = {
                "std": float(np.std(window)),
                "range": float(np.ptp(window)),
                "diff_std": float(np.std(np.diff(window))),
                "diff_max": float(np.max(np.abs(np.diff(window)))),
                "kurtosis": float(pd.Series(window).kurtosis()),
            }

            if mask_window.mean() > 0.5:  # 多数被标为异常
                anomaly_features.append(features)
            elif mask_window.mean() < 0.1:  # 大部分正常
                normal_features.append(features)

    if not anomaly_features or not normal_features:
        return {}

    # 学习阈值：取异常窗口特征的中位数作为阈值
    thresholds = {}
    for key in anomaly_features[0].keys():
        a_vals = [f[key] for f in anomaly_features]
        n_vals = [f[key] for f in normal_features]
        # 阈值 = 正常窗口的 95 百分位
        thresholds[key] = {
            "threshold": float(np.percentile(n_vals, 95)),
            "anomaly_median": float(np.median(a_vals)),
            "normal_median": float(np.median(n_vals)),
            "separation": float(np.median(a_vals) - np.median(n_vals)),
        }

    print(f"[Teacher-Student] 学习完成: {len(anomaly_features)} 异常窗口, "
          f"{len(normal_features)} 正常窗口")
    for k, v in thresholds.items():
        print(f"  {k}: threshold={v['threshold']:.4f}, "
              f"sep={v['separation']:.4f}")

    return thresholds


def student_detect(values: np.ndarray, thresholds: dict,
                    window_size: int = 64, min_features: int = 2) -> np.ndarray:
    """Student 模型：用学到的阈值检测异常"""
    n = len(values)
    mean_v, std_v = values.mean(), values.std()
    if std_v < 1e-10:
        return np.zeros(n, dtype=int)
    norm_v = (values - mean_v) / std_v

    mask = np.zeros(n, dtype=int)

    for start in range(0, n - window_size, window_size // 4):
        window = norm_v[start:start + window_size]
        diff = np.diff(window)

        features = {
            "std": float(np.std(window)),
            "range": float(np.ptp(window)),
            "diff_std": float(np.std(diff)),
            "diff_max": float(np.max(np.abs(diff))),
            "kurtosis": float(pd.Series(window).kurtosis()),
        }

        # 统计超过阈值的特征数量
        exceeded = 0
        for key, feat_val in features.items():
            if key in thresholds and feat_val > thresholds[key]["threshold"]:
                exceeded += 1

        if exceeded >= min_features:
            mask[start:start + window_size] = 1

    return mask


def main():
    data_dir = str(PROJECT_ROOT / "data/adtk_hbos_old")
    teacher_dir = str(PROJECT_ROOT / "results/predictions/timer")
    out_dir = str(PROJECT_ROOT / "results/predictions/teacher_student")
    os.makedirs(out_dir, exist_ok=True)

    with open(PROJECT_ROOT / "data/cleaned/evaluated_points.txt") as f:
        test_points = [l.strip() for l in f if l.strip()]

    # Step 1: 从 Teacher 学习阈值
    print("[Teacher-Student] 从 Timer 伪标签学习...")
    thresholds = learn_from_teacher(data_dir, teacher_dir, test_points)

    if not thresholds:
        print("ERROR: 无法学习阈值")
        return

    # Step 2: Student 推理
    print(f"\n[Teacher-Student] Student 检测 {len(test_points)} 点位...")
    success = 0

    for pt in test_points:
        matches = glob.glob(os.path.join(data_dir,
                    f"global_adtk_hbos_*_{pt}_*_trend_seasonal_resid.csv"))
        if not matches:
            continue

        try:
            df = pd.read_csv(matches[0])
            skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                    "global_mask_cluster", "local_mask_cluster"}
            vcol = [c for c in df.columns if c not in skip][0]
            v = df[vcol].values.astype(np.float64)
            v[np.isnan(v)] = np.nanmedian(v) if not np.all(np.isnan(v)) else 0
            orig = len(v)

            if orig > 5000:
                ds_v, ds_idx = downsample_m4(v, 5000)
            else:
                ds_v, ds_idx = v, np.arange(orig)

            mask_ds = student_detect(ds_v, thresholds)

            if len(ds_v) < orig:
                mask = np.zeros(orig, dtype=int)
                for i, idx in enumerate(ds_idx):
                    if mask_ds[i] == 1:
                        mask[idx] = 1
            else:
                mask = mask_ds

            rate = float(mask.mean())
            pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(
                os.path.join(out_dir, f"{pt}.csv"), index=False)
            with open(os.path.join(out_dir, f"{pt}.status.json"), "w") as f:
                json.dump({"status": "success", "anomaly_rate": rate,
                           "anomaly_count": int(mask.sum()),
                           "total_rows": orig}, f, indent=2)
            success += 1
        except:
            pass

    print(f"[Teacher-Student] 完成: {success}/{len(test_points)}")
    files = glob.glob(os.path.join(out_dir, "*.status.json"))
    rates = []
    for f in files:
        d = json.load(open(f))
        if d.get("status") == "success":
            rates.append(d["anomaly_rate"])
    if rates:
        print(f"  mean_rate={np.mean(rates):.4f}, median={np.median(rates):.4f}")

    # 保存学到的阈值
    with open(os.path.join(out_dir, "learned_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)


if __name__ == "__main__":
    main()

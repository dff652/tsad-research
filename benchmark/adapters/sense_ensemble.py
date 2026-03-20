#!/usr/bin/env python3
"""
SENSE：选择性集成异常检测（参考 TSB-AutoAD VLDB 2025）

核心思想：
- 不是简单的投票/加权，而是"根据数据特征选择最优模型"
- 对每个点位，根据其时序特征选择最适合的算法
- 然后用选中的几个算法做加权集成

本实现：
1. 提取时序统计特征（均值、方差、自相关、趋势、周期性等）
2. 根据特征匹配最适合的算法组合
3. 加权融合选中算法的检测结果

路由规则（基于本项目的实验发现）：
- 高方差/多跳变 → Timer + Wavelet（互补）
- 低方差/平稳 → IForest（保守检测）
- 高异常率传感器 → Timer (k=4.5)
- 低异常率传感器 → Timer (k=2.5) + MAD
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_ts_features(values: np.ndarray) -> dict:
    """提取时序统计特征用于路由"""
    n = len(values)
    features = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "cv": float(np.std(values) / max(abs(np.mean(values)), 1e-10)),
        "range": float(np.ptp(values)),
        "skewness": float(pd.Series(values).skew()),
        "kurtosis": float(pd.Series(values).kurtosis()),
    }

    # 差分特征
    diff = np.diff(values)
    features["diff_std"] = float(np.std(diff))
    features["diff_max"] = float(np.max(np.abs(diff)))

    # 跳变数量（差分超过 3 倍标准差）
    diff_threshold = np.std(diff) * 3
    features["jump_count"] = int(np.sum(np.abs(diff) > diff_threshold)) if diff_threshold > 0 else 0

    # 自相关（滞后 1）
    if n > 10:
        features["autocorr_1"] = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    else:
        features["autocorr_1"] = 0.0

    return features


def route_to_algorithms(features: dict) -> list:
    """根据特征路由到最优算法组合"""
    algorithms = []

    cv = features["cv"]
    jump_count = features["jump_count"]
    autocorr = features["autocorr_1"]
    kurtosis = features["kurtosis"]

    # 规则 1：高变异系数或多跳变 → Timer + Wavelet
    if cv > 0.3 or jump_count > 10:
        algorithms.append(("timer", 3.0))
        algorithms.append(("wavelet_sensitive", 1.5))
        algorithms.append(("iforest", 1.0))

    # 规则 2：高峰度（有尖峰） → IForest + MAD
    elif kurtosis > 5:
        algorithms.append(("timer", 2.5))
        algorithms.append(("iforest", 2.0))
        algorithms.append(("mad", 1.0))

    # 规则 3：高自相关（平稳）→ Timer 为主
    elif autocorr > 0.95:
        algorithms.append(("timer", 3.0))
        algorithms.append(("freq_patch", 1.0))

    # 规则 4：默认组合
    else:
        algorithms.append(("timer", 3.0))
        algorithms.append(("iforest", 1.5))
        algorithms.append(("wavelet_sensitive", 1.0))

    return algorithms


def load_prediction(algo_name: str, point_name: str) -> np.ndarray:
    """加载算法预测结果"""
    pred_dir = PROJECT_ROOT / "results/predictions" / algo_name
    csv_path = pred_dir / f"{point_name}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "global_mask" in df.columns:
        return df["global_mask"].values.astype(int)
    return None


def sense_detect(point_name: str, values: np.ndarray) -> tuple:
    """SENSE 选择性集成检测"""
    features = extract_ts_features(values)
    algo_weights = route_to_algorithms(features)

    masks = []
    weights = []

    for algo_name, weight in algo_weights:
        mask = load_prediction(algo_name, point_name)
        if mask is not None:
            masks.append(mask)
            weights.append(weight)

    if not masks:
        return np.zeros(len(values), dtype=int), features, []

    # 对齐长度
    min_len = min(len(m) for m in masks)
    aligned = [m[:min_len] for m in masks]

    # 加权融合
    total_weight = sum(weights)
    weighted = np.zeros(min_len)
    for m, w in zip(aligned, weights):
        weighted += m.astype(float) * (w / total_weight)

    # 阈值 0.3（至少约 1/3 的加权投票）
    result = (weighted >= 0.3).astype(int)

    # 补齐到原始长度
    if min_len < len(values):
        full = np.zeros(len(values), dtype=int)
        full[:min_len] = result
        result = full

    selected = [a[0] for a in algo_weights if load_prediction(a[0], point_name) is not None]
    return result, features, selected


def main():
    data_dir = str(PROJECT_ROOT / "data/adtk_hbos_old")
    out_dir = str(PROJECT_ROOT / "results/predictions/sense")
    os.makedirs(out_dir, exist_ok=True)

    with open(PROJECT_ROOT / "data/cleaned/evaluated_points.txt") as f:
        points = [l.strip() for l in f if l.strip()]

    print(f"[SENSE] 选择性集成: {len(points)} 点位")
    success = 0

    for i, pt in enumerate(points):
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

            mask, features, selected = sense_detect(pt, v)
            rate = float(mask.mean())

            pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(
                os.path.join(out_dir, f"{pt}.csv"), index=False)
            with open(os.path.join(out_dir, f"{pt}.status.json"), "w") as f:
                json.dump({
                    "status": "success", "anomaly_rate": rate,
                    "anomaly_count": int(mask.sum()), "total_rows": len(v),
                    "selected_algorithms": selected,
                    "features": {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in features.items()},
                }, f, indent=2)
            success += 1
        except Exception as e:
            pass

    print(f"[SENSE] 完成: {success}/{len(points)}")
    files = glob.glob(os.path.join(out_dir, "*.status.json"))
    rates = []
    for f in files:
        d = json.load(open(f))
        if d.get("status") == "success":
            rates.append(d["anomaly_rate"])
    if rates:
        print(f"  mean_rate={np.mean(rates):.4f}, median={np.median(rates):.4f}")


if __name__ == "__main__":
    main()

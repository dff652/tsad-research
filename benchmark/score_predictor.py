#!/usr/bin/env python3
"""
里程碑 2：量化评分预测器

核心思路：将人工评分的隐含逻辑自动化
- 分析 ADTK+HBOS 推理特征与 Timer/ChatTS/Qwen 人工评分的关系
- 建立"物理特征 → 预期评分"的量化模型
- 用于预测新算法可能获得的人工评分

关键指标：
1. 异常率与评分的非线性关系（过高过低都差）
2. 按传感器类型分层的差异化阈值
3. 评审一致性加权（高一致性的点位权重更大）
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path


def load_merged_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def analyze_score_drivers(df: pd.DataFrame) -> dict:
    """分析各特征与人工评分的关系"""
    features = [
        "global_mask_ratio", "num_anomaly_clusters", "avg_cluster_length",
        "label_agreement_rate", "sensor_cv", "sensor_range",
    ]
    models = ["avg_qwen", "avg_chatts", "avg_timer"]

    correlations = {}
    for model in models:
        if model not in df.columns:
            continue
        model_corr = {}
        for feat in features:
            if feat in df.columns:
                valid = df[[feat, model]].dropna()
                if len(valid) > 5:
                    model_corr[feat] = float(valid[feat].corr(valid[model]))
        correlations[model] = model_corr

    return correlations


def build_anomaly_rate_bins(df: pd.DataFrame) -> dict:
    """按异常率分箱统计各模型评分"""
    bins = [0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 1.01]
    labels = ["0-0.1%", "0.1-0.5%", "0.5-1%", "1-5%", "5-10%", "10-20%", "20%+"]

    df = df.copy()
    df["rate_bin"] = pd.cut(df["global_mask_ratio"], bins=bins, labels=labels, right=False)

    models = ["avg_qwen", "avg_chatts", "avg_timer"]
    result = {}
    for model in models:
        if model not in df.columns:
            continue
        bin_stats = df.groupby("rate_bin", observed=True)[model].agg(["mean", "count", "std"])
        result[model] = bin_stats.to_dict("index")

    return result


def build_sensor_type_profiles(df: pd.DataFrame) -> dict:
    """按传感器类型构建差异化评估配置"""
    if "sensor_type" not in df.columns:
        return {}

    models = ["avg_qwen", "avg_chatts", "avg_timer"]
    profiles = {}

    for stype, group in df.groupby("sensor_type"):
        if len(group) < 3:
            continue
        profile = {
            "count": len(group),
            "mean_anomaly_rate": float(group["global_mask_ratio"].mean()),
            "std_anomaly_rate": float(group["global_mask_ratio"].std()),
        }
        for model in models:
            if model in group.columns:
                profile[f"{model}_mean"] = float(group[model].mean())
                profile[f"{model}_std"] = float(group[model].std())
        profiles[stype] = profile

    return profiles


def compute_evaluation_thresholds(df: pd.DataFrame) -> dict:
    """从 Timer 评分数据反推最优异常检测阈值区间

    思路：Timer 评分最高的点位，其 ADTK 异常检测特征表现如何？
    """
    if "avg_timer" not in df.columns:
        return {}

    # Timer 高分组 (≥0.8) vs 低分组 (<0.3)
    high = df[df["avg_timer"] >= 0.8]
    low = df[df["avg_timer"] < 0.3]

    features = ["global_mask_ratio", "num_anomaly_clusters", "avg_cluster_length",
                "sensor_cv", "label_agreement_rate"]

    thresholds = {}
    for feat in features:
        if feat not in df.columns:
            continue
        if len(high) > 0 and len(low) > 0:
            thresholds[feat] = {
                "high_score_mean": float(high[feat].mean()),
                "high_score_median": float(high[feat].median()),
                "low_score_mean": float(low[feat].mean()),
                "low_score_median": float(low[feat].median()),
                "separation": float(high[feat].mean() - low[feat].mean()),
            }

    return thresholds


def generate_full_report(merged_csv: str, output_dir: str):
    """生成完整的量化评估报告"""
    df = load_merged_data(merged_csv)
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载 {len(df)} 条评分-特征关联数据")

    # 1. 相关性分析
    corr = analyze_score_drivers(df)
    print("\n=== 特征-评分相关性 ===")
    for model, feats in corr.items():
        print(f"\n{model}:")
        for feat, r in sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feat:<25s} r={r:+.4f}")

    # 2. 异常率分箱分析
    bins = build_anomaly_rate_bins(df)
    print("\n=== 异常率分箱 → 评分 ===")
    for model, bin_data in bins.items():
        print(f"\n{model}:")
        for bin_label, stats in bin_data.items():
            if stats["count"] > 0:
                mean = stats["mean"]
                print(f"  {bin_label:<12s} mean={mean:.3f} (n={stats['count']:.0f})")

    # 3. 传感器类型 Profile
    profiles = build_sensor_type_profiles(df)
    print("\n=== 传感器类型 Profile ===")
    for stype, prof in sorted(profiles.items(), key=lambda x: -x[1]["count"]):
        timer_score = prof.get("avg_timer_mean", 0)
        print(f"  {stype:<25s} n={prof['count']:>3d}  "
              f"rate={prof['mean_anomaly_rate']:.4f}  "
              f"timer_score={timer_score:.3f}")

    # 4. 最优阈值反推
    thresholds = compute_evaluation_thresholds(df)
    print("\n=== Timer 高分 vs 低分特征对比 ===")
    for feat, vals in thresholds.items():
        print(f"  {feat:<25s} high={vals['high_score_mean']:.4f}  "
              f"low={vals['low_score_mean']:.4f}  "
              f"Δ={vals['separation']:+.4f}")

    # 保存报告
    report = {
        "correlations": corr,
        "anomaly_rate_bins": bins,
        "sensor_type_profiles": profiles,
        "evaluation_thresholds": thresholds,
    }
    report_path = os.path.join(output_dir, "score_prediction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    merged_csv = str(PROJECT_ROOT / "data/features/merged_scores_features.csv")
    output_dir = str(PROJECT_ROOT / "results")
    generate_full_report(merged_csv, output_dir)

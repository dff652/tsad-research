#!/usr/bin/env python3
"""
Timer 检测结果与人工评分的直接关联分析

将 Timer 的实际推理特征（异常率、簇特征等）与 5 位专家的评分进行关联，
找出决定评分高低的关键因素。
"""

import json
import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_timer_features(point_name: str) -> dict:
    """提取 Timer 推理结果的特征"""
    csv_path = PROJECT_ROOT / f"results/predictions/timer/{point_name}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    mask = df["global_mask"].values.astype(int)

    # 基本统计
    anomaly_count = int(mask.sum())
    total = len(mask)
    rate = mask.mean()

    # 簇统计
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts if len(starts) > 0 else np.array([])

    return {
        "timer_anomaly_rate": float(rate),
        "timer_anomaly_count": anomaly_count,
        "timer_num_clusters": len(lengths),
        "timer_avg_cluster_len": float(np.mean(lengths)) if len(lengths) > 0 else 0,
        "timer_max_cluster_len": int(np.max(lengths)) if len(lengths) > 0 else 0,
        "timer_jump_ratio": float(2 * len(lengths) / anomaly_count) if anomaly_count > 0 else 0,
    }


def main():
    # 加载评分数据
    scores = pd.read_csv(PROJECT_ROOT / "data/features/merged_scores_features.csv")

    # 提取 Timer 推理特征
    timer_features = []
    for _, row in scores.iterrows():
        point = row["point_name"]
        feat = extract_timer_features(point)
        if feat:
            feat["point_name"] = point
            timer_features.append(feat)

    timer_df = pd.DataFrame(timer_features)
    print(f"匹配到 {len(timer_df)} 个有 Timer 推理结果的评分点位")

    # 合并
    merged = scores.merge(timer_df, on="point_name", how="inner")
    print(f"合并后: {len(merged)} 条数据")

    # 1. Timer 推理特征与 Timer 人工评分的相关性
    print("\n=== Timer 推理特征 vs Timer 人工评分 (avg_timer) ===")
    timer_feats = [c for c in merged.columns if c.startswith("timer_") and c != "avg_timer"
                   and "dff_" not in c and "wyx_" not in c and "lzh_" not in c
                   and "xpj_" not in c and "aym_" not in c]
    for feat in timer_feats:
        r = merged[feat].corr(merged["avg_timer"])
        print(f"  {feat:<30} r={r:+.4f}")

    # 2. Timer 推理特征 vs ADTK 特征的关系
    print("\n=== Timer 推理特征 vs ADTK 特征 ===")
    adtk_timer_pairs = [
        ("timer_anomaly_rate", "global_mask_ratio"),
        ("timer_num_clusters", "num_anomaly_clusters"),
        ("timer_avg_cluster_len", "avg_cluster_length"),
    ]
    for t_feat, a_feat in adtk_timer_pairs:
        if t_feat in merged.columns and a_feat in merged.columns:
            r = merged[t_feat].corr(merged[a_feat])
            print(f"  {t_feat} vs {a_feat}: r={r:+.4f}")

    # 3. Timer 评分分层分析
    print("\n=== Timer 评分分层分析 ===")
    high = merged[merged["avg_timer"] >= 0.8]
    mid = merged[(merged["avg_timer"] >= 0.4) & (merged["avg_timer"] < 0.8)]
    low = merged[merged["avg_timer"] < 0.4]

    for label, group in [("高分(≥0.8)", high), ("中分(0.4-0.8)", mid), ("低分(<0.4)", low)]:
        if len(group) == 0:
            continue
        print(f"\n  {label} (n={len(group)}):")
        print(f"    Timer推理 异常率: mean={group['timer_anomaly_rate'].mean():.4f}, "
              f"median={group['timer_anomaly_rate'].median():.4f}")
        print(f"    Timer推理 簇数量: mean={group['timer_num_clusters'].mean():.1f}")
        print(f"    Timer推理 簇长度: mean={group['timer_avg_cluster_len'].mean():.1f}")
        print(f"    ADTK 异常率:     mean={group['global_mask_ratio'].mean():.4f}")
        print(f"    ADTK 簇数量:     mean={group['num_anomaly_clusters'].mean():.1f}")

    # 4. 按评价类别分析
    if "eval_category" in merged.columns:
        print("\n=== 按评价类别分析 Timer 推理特征 ===")
        for cat, group in merged.groupby("eval_category"):
            if len(group) < 2:
                continue
            print(f"\n  {cat} (n={len(group)}):")
            print(f"    Timer推理: rate={group['timer_anomaly_rate'].mean():.4f}, "
                  f"clusters={group['timer_num_clusters'].mean():.0f}")
            print(f"    人工评分: timer={group['avg_timer'].mean():.3f}, "
                  f"chatts={group['avg_chatts'].mean():.3f}")

    # 5. Timer 推理 vs ADTK 的异常率偏差分析
    print("\n=== Timer 推理 vs ADTK 异常率偏差 ===")
    merged["rate_diff"] = merged["timer_anomaly_rate"] - merged["global_mask_ratio"]
    merged["rate_ratio"] = merged["timer_anomaly_rate"] / (merged["global_mask_ratio"] + 1e-8)

    print(f"  Timer > ADTK: {(merged['rate_diff'] > 0).sum()} 点位")
    print(f"  Timer < ADTK: {(merged['rate_diff'] < 0).sum()} 点位")
    print(f"  平均偏差: {merged['rate_diff'].mean():.4f}")
    print(f"  偏差与 Timer 评分相关: r={merged['rate_diff'].corr(merged['avg_timer']):+.4f}")

    # 保存完整分析结果
    analysis = {
        "timer_feature_correlations": {
            feat: float(merged[feat].corr(merged["avg_timer"])) for feat in timer_feats
        },
        "adtk_timer_correlations": {
            f"{t} vs {a}": float(merged[t].corr(merged[a]))
            for t, a in adtk_timer_pairs if t in merged.columns and a in merged.columns
        },
        "score_stratification": {
            "high_mean_rate": float(high["timer_anomaly_rate"].mean()) if len(high) > 0 else None,
            "low_mean_rate": float(low["timer_anomaly_rate"].mean()) if len(low) > 0 else None,
        },
        "rate_diff_correlation": float(merged["rate_diff"].corr(merged["avg_timer"])),
    }
    output_path = PROJECT_ROOT / "results/timer_score_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n分析结果已保存: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
自动评分器 —— 基于算法检测特征预测人工评分

由于人工评分资源有限（仅 100 个点位 × 5 人），本模块尝试：
1. 从已有的 {Timer评分, ADTK特征} 数据中学习评分模式
2. 对新算法的检测结果自动预测"可能的人工评分"
3. 为全量 436 点位提供弱监督质量评估

方法：
- 特征空间：异常率、簇数量、簇长度、跳变率等
- 目标：Timer 的人工均分 (0~1)
- 模型：简单决策树/分箱规则（可解释性优先）
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_training_data() -> pd.DataFrame:
    """加载训练数据：已评分点位的特征 + Timer 评分"""
    merged = pd.read_csv(PROJECT_ROOT / "data/features/merged_scores_features.csv")
    return merged


def extract_prediction_features(algo_name: str, point_name: str) -> dict:
    """从算法预测结果中提取特征"""
    csv_path = PROJECT_ROOT / f"results/predictions/{algo_name}/{point_name}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if "global_mask" not in df.columns:
        return None

    mask = df["global_mask"].values.astype(int)
    total = len(mask)
    anomaly_count = int(mask.sum())
    anomaly_rate = mask.mean()

    # 簇统计
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts if len(starts) > 0 else np.array([])

    return {
        "point_name": point_name,
        "anomaly_rate": float(anomaly_rate),
        "anomaly_count": anomaly_count,
        "total_rows": total,
        "num_clusters": len(lengths),
        "avg_cluster_len": float(np.mean(lengths)) if len(lengths) > 0 else 0,
        "max_cluster_len": int(np.max(lengths)) if len(lengths) > 0 else 0,
        "jump_ratio": float(2 * len(lengths) / anomaly_count) if anomaly_count > 0 else 0,
    }


def build_scoring_rules(training_df: pd.DataFrame) -> dict:
    """从训练数据中构建评分规则

    基于 Timer 评分数据的统计分析，建立可解释的分箱规则。
    """
    # Timer 高分(≥0.8)和低分(<0.3)的特征对比
    high = training_df[training_df["avg_timer"] >= 0.8]
    low = training_df[training_df["avg_timer"] < 0.3]

    rules = {
        "high_score_profile": {
            "anomaly_rate_range": [
                float(high["global_mask_ratio"].quantile(0.1)),
                float(high["global_mask_ratio"].quantile(0.9)),
            ],
            "cluster_count_range": [
                float(high["num_anomaly_clusters"].quantile(0.1)),
                float(high["num_anomaly_clusters"].quantile(0.9)),
            ],
        },
        "low_score_indicators": {
            "zero_anomaly": True,  # 无异常 = 差评（漏检）
            "extreme_rate": 0.5,   # 异常率 > 50% = 差评（乱叫）
        },
        "thresholds": {
            "anomaly_rate_sweet_spot": [0.001, 0.15],  # 合理异常率区间
        },
    }

    return rules


def predict_score(features: dict, rules: dict) -> float:
    """基于规则预测评分 (0~1)"""
    rate = features["anomaly_rate"]
    clusters = features["num_clusters"]

    # 规则 1：零异常 = 差评
    if rate == 0:
        return 0.2

    # 规则 2：极端高异常率 = 差评
    if rate > 0.5:
        return 0.1

    # 规则 3：异常率在合理区间内
    sweet = rules["thresholds"]["anomaly_rate_sweet_spot"]
    if sweet[0] <= rate <= sweet[1]:
        base_score = 0.65
    elif rate < sweet[0]:
        base_score = 0.4  # 检测不足
    else:
        base_score = 0.3  # 过度检测

    # 调节 4：簇数量奖励（更多精细簇 = 更好的检测）
    if clusters > 50:
        base_score += 0.1
    elif clusters > 10:
        base_score += 0.05

    # 调节 5：适中的簇长度奖励
    avg_len = features["avg_cluster_len"]
    if 10 < avg_len < 5000:
        base_score += 0.05

    return min(1.0, max(0.0, base_score))


def score_algorithm(algo_name: str, points: list = None) -> dict:
    """对一个算法的所有预测结果进行自动评分"""
    training_df = load_training_data()
    rules = build_scoring_rules(training_df)

    if points is None:
        with open(PROJECT_ROOT / "data/cleaned/evaluated_points.txt") as f:
            points = [l.strip() for l in f if l.strip()]

    scores = []
    for point in points:
        features = extract_prediction_features(algo_name, point)
        if features is None:
            continue
        score = predict_score(features, rules)
        features["predicted_score"] = score
        scores.append(features)

    if not scores:
        return {"algorithm": algo_name, "error": "no predictions"}

    pred_scores = [s["predicted_score"] for s in scores]
    return {
        "algorithm": algo_name,
        "scored_points": len(scores),
        "mean_predicted_score": float(np.mean(pred_scores)),
        "median_predicted_score": float(np.median(pred_scores)),
        "pass_rate": float(np.mean([1 if s >= 0.5 else 0 for s in pred_scores])),
        "good_rate": float(np.mean([1 if s >= 0.8 else 0 for s in pred_scores])),
    }


def main():
    """对所有有预测结果的算法进行自动评分"""
    pred_base = PROJECT_ROOT / "results/predictions"
    algos = [d.name for d in pred_base.iterdir() if d.is_dir()]

    print(f"自动评分: {len(algos)} 个算法\n")

    all_scores = {}
    for algo in sorted(algos):
        result = score_algorithm(algo)
        if "error" in result:
            continue
        all_scores[algo] = result
        print(f"{algo:<20} n={result['scored_points']:>3d}  "
              f"pred_score={result['mean_predicted_score']:.3f}  "
              f"pass={result['pass_rate']:.1%}  "
              f"good={result['good_rate']:.1%}")

    # 保存
    output_path = PROJECT_ROOT / "results/auto_scores.json"
    with open(output_path, "w") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    # 与实际 Timer 评分对比
    training_df = load_training_data()
    if "avg_timer" in training_df.columns:
        real_score = training_df["avg_timer"].mean()
        print(f"\n实际 Timer 人工均分: {real_score:.3f}")
        if "timer" in all_scores:
            pred = all_scores["timer"]["mean_predicted_score"]
            print(f"自动预测 Timer 均分: {pred:.3f}")
            print(f"偏差: {abs(pred - real_score):.3f}")


if __name__ == "__main__":
    main()

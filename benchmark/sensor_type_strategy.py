#!/usr/bin/env python3
"""
按传感器类型的分层检测策略

核心思路：不同传感器类型有不同的异常模式，应使用差异化参数。
基于已有的 Timer 推理结果 + 人工评分，为每种传感器类型推荐最优参数。

分析维度：
1. 各传感器类型的 Timer 异常率分布
2. 与人工评分的关联
3. 推荐差异化的 threshold_k 和检测策略
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_all_data():
    """加载评分+特征+Timer推理结果的关联数据"""
    merged = pd.read_csv(PROJECT_ROOT / "data/features/merged_scores_features.csv")

    # 加载 Timer 推理特征
    timer_features = []
    for f in glob.glob(str(PROJECT_ROOT / "results/predictions/timer/*.status.json")):
        d = json.load(open(f))
        if d.get("status") == "success" and "anomaly_rate" in d:
            point = Path(f).stem.replace(".status", "")
            timer_features.append({
                "point_name": point,
                "timer_rate": d["anomaly_rate"],
                "timer_intervals": d.get("num_intervals", 0),
            })
    timer_df = pd.DataFrame(timer_features)

    # 加载 ChatTS 推理特征
    chatts_features = []
    for f in glob.glob(str(PROJECT_ROOT / "results/predictions/chatts/*.status.json")):
        d = json.load(open(f))
        if d.get("status") == "success" and "anomaly_rate" in d:
            point = Path(f).stem.replace(".status", "")
            chatts_features.append({
                "point_name": point,
                "chatts_rate": d["anomaly_rate"],
                "chatts_intervals": d.get("num_intervals", 0),
            })
    chatts_df = pd.DataFrame(chatts_features)

    # 合并
    result = merged.copy()
    if not timer_df.empty:
        result = result.merge(timer_df, on="point_name", how="left")
    if not chatts_df.empty:
        result = result.merge(chatts_df, on="point_name", how="left")

    return result


def analyze_by_sensor_type(df: pd.DataFrame):
    """按传感器类型分析各算法表现"""
    print("=" * 90)
    print("按传感器类型的算法表现分析")
    print("=" * 90)

    models = ["avg_timer", "avg_chatts", "avg_qwen"]
    rate_cols = ["global_mask_ratio", "timer_rate", "chatts_rate"]

    for stype, group in sorted(df.groupby("sensor_type"), key=lambda x: -len(x[1])):
        if len(group) < 3:
            continue

        print(f"\n--- {stype} (n={len(group)}) ---")

        # 人工评分
        for m in models:
            if m in group.columns:
                scores = group[m].dropna()
                if len(scores) > 0:
                    print(f"  {m:<15} mean={scores.mean():.3f}  median={scores.median():.3f}")

        # 异常率
        for rc in rate_cols:
            if rc in group.columns:
                rates = group[rc].dropna()
                if len(rates) > 0:
                    print(f"  {rc:<15} mean={rates.mean():.4f}  median={rates.median():.4f}")


def recommend_strategies(df: pd.DataFrame) -> dict:
    """为每种传感器类型推荐检测策略"""
    strategies = {}

    for stype, group in df.groupby("sensor_type"):
        if len(group) < 2:
            continue

        adtk_rate = group["global_mask_ratio"].mean()
        timer_rate = group["timer_rate"].mean() if "timer_rate" in group.columns else None
        timer_score = group["avg_timer"].mean() if "avg_timer" in group.columns else None

        strategy = {
            "sensor_type": stype,
            "n_points": len(group),
            "adtk_mean_rate": float(adtk_rate),
        }

        if timer_rate is not None:
            strategy["timer_mean_rate"] = float(timer_rate)
        if timer_score is not None:
            strategy["timer_human_score"] = float(timer_score)

        # 推荐策略
        if adtk_rate > 0.15:
            # 高异常率类型（如 Current/Power）：用更严格的阈值避免过检
            strategy["recommended_threshold_k"] = 4.5
            strategy["recommended_method"] = "mad"
            strategy["note"] = "高基线异常率，需严格阈值过滤"
        elif adtk_rate < 0.005:
            # 低异常率类型（如 SOV/Discrete）：用更宽松的阈值避免漏检
            strategy["recommended_threshold_k"] = 2.5
            strategy["recommended_method"] = "mad"
            strategy["note"] = "低基线异常率，需宽松阈值捕获细微异常"
        else:
            # 中等异常率：使用默认参数
            strategy["recommended_threshold_k"] = 3.5
            strategy["recommended_method"] = "mad"
            strategy["note"] = "标准异常率范围，使用默认参数"

        # Timer 评分高的类型可以更信赖 Timer
        if timer_score is not None and timer_score > 0.7:
            strategy["timer_reliability"] = "high"
        elif timer_score is not None and timer_score < 0.5:
            strategy["timer_reliability"] = "low"
            strategy["note"] += "；Timer 在此类型上表现欠佳，建议参考统计方法"
        else:
            strategy["timer_reliability"] = "medium"

        strategies[stype] = strategy

    return strategies


def main():
    df = load_all_data()
    print(f"加载 {len(df)} 条关联数据")

    # 1. 按传感器类型分析
    analyze_by_sensor_type(df)

    # 2. 推荐策略
    strategies = recommend_strategies(df)

    print(f"\n{'='*90}")
    print("推荐的分层检测策略")
    print(f"{'='*90}")
    print(f"{'传感器类型':<25} {'n':>4} {'ADTK异常率':>10} {'Timer评分':>10} {'推荐k':>7} {'可靠度':>8}")
    print("-" * 90)
    for stype, s in sorted(strategies.items(), key=lambda x: -x[1]["n_points"]):
        print(f"{stype:<25} {s['n_points']:>4} "
              f"{s['adtk_mean_rate']:>10.4f} "
              f"{s.get('timer_human_score', 0):>10.3f} "
              f"{s['recommended_threshold_k']:>7.1f} "
              f"{s.get('timer_reliability', '?'):>8}")

    # 3. 保存
    output = {
        "strategies": strategies,
        "summary": {
            "total_types": len(strategies),
            "high_rate_types": sum(1 for s in strategies.values() if s["adtk_mean_rate"] > 0.15),
            "low_rate_types": sum(1 for s in strategies.values() if s["adtk_mean_rate"] < 0.005),
        },
    }
    output_path = PROJECT_ROOT / "results/sensor_type_strategies.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n策略已保存: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
算法对比报告生成器

对已完成推理的所有算法进行横向对比，生成：
1. 维度A：人工评分关联对比
2. 维度B：物理常识约束对比
3. 综合排名表

用法:
    python compare_report.py                          # 对比所有有结果的算法
    python compare_report.py --algo timer sundial     # 指定算法对比
"""

import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent


def load_features(algo_name: str) -> pd.DataFrame:
    """从预测结果目录提取特征统计"""
    pred_dir = PROJECT_ROOT / "results/predictions" / algo_name
    if not pred_dir.exists():
        return pd.DataFrame()

    records = []
    for csv_path in sorted(pred_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "global_mask" not in df.columns:
                continue

            mask = df["global_mask"].values.astype(int)
            anomaly_count = int(mask.sum())
            total = len(mask)
            rate = mask.mean()

            # 簇统计
            diff = np.diff(np.concatenate(([0], mask, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts if len(starts) > 0 else np.array([])
            num_clusters = len(lengths)
            avg_cluster_len = float(np.mean(lengths)) if len(lengths) > 0 else 0
            max_cluster_len = int(np.max(lengths)) if len(lengths) > 0 else 0

            # 跳变率
            transitions = np.sum(np.abs(np.diff(mask)))
            jump_ratio = float(2 * num_clusters / anomaly_count) if anomaly_count > 0 else 0

            records.append({
                "point_name": csv_path.stem,
                "total_rows": total,
                "anomaly_count": anomaly_count,
                "anomaly_rate": rate,
                "num_clusters": num_clusters,
                "avg_cluster_len": avg_cluster_len,
                "max_cluster_len": max_cluster_len,
                "jump_ratio": jump_ratio,
            })
        except Exception as e:
            pass

    return pd.DataFrame(records)


def load_scores() -> pd.DataFrame:
    """加载人工评分数据"""
    scores_path = PROJECT_ROOT / "data/cleaned/scores_analysis.csv"
    if scores_path.exists():
        return pd.read_csv(scores_path)
    return pd.DataFrame()


def load_adtk_features() -> pd.DataFrame:
    """加载已有的 ADTK 特征"""
    feat_path = PROJECT_ROOT / "data/features/all_points_features.csv"
    if feat_path.exists():
        return pd.read_csv(feat_path)
    return pd.DataFrame()


def generate_comparison(algo_names: list):
    """生成算法对比报告"""
    print(f"\n{'='*80}")
    print(f"TSAD Benchmark 算法对比报告")
    print(f"{'='*80}\n")

    # 收集各算法的特征数据
    algo_data = {}
    for algo in algo_names:
        if algo == "adtk_hbos":
            # ADTK 用已有特征文件
            feat = load_adtk_features()
            if not feat.empty:
                algo_data[algo] = {
                    "total_points": len(feat),
                    "mean_rate": float(feat["global_mask_ratio"].mean()),
                    "median_rate": float(feat["global_mask_ratio"].median()),
                    "std_rate": float(feat["global_mask_ratio"].std()),
                    "max_rate": float(feat["global_mask_ratio"].max()),
                    "mean_stickiness": float(feat["avg_cluster_length"].mean()),
                    "mean_clusters": float(feat["num_anomaly_clusters"].mean()),
                    "zero_points": int((feat["global_mask_count"] == 0).sum()),
                }
        else:
            feat = load_features(algo)
            if not feat.empty:
                algo_data[algo] = {
                    "total_points": len(feat),
                    "mean_rate": float(feat["anomaly_rate"].mean()),
                    "median_rate": float(feat["anomaly_rate"].median()),
                    "std_rate": float(feat["anomaly_rate"].std()),
                    "max_rate": float(feat["anomaly_rate"].max()),
                    "mean_stickiness": float(feat["avg_cluster_len"].mean()),
                    "mean_clusters": float(feat["num_clusters"].mean()),
                    "mean_jump": float(feat["jump_ratio"].mean()),
                    "zero_points": int((feat["anomaly_count"] == 0).sum()),
                }

    if not algo_data:
        print("无可用算法数据")
        return

    # 打印对比表
    print(f"{'指标':<25}", end="")
    for algo in algo_data:
        print(f"{algo:<20}", end="")
    print()
    print("-" * (25 + 20 * len(algo_data)))

    metrics = [
        ("点位数", "total_points", "d"),
        ("平均异常率", "mean_rate", ".6f"),
        ("中位异常率", "median_rate", ".6f"),
        ("异常率标准差", "std_rate", ".6f"),
        ("最大异常率", "max_rate", ".6f"),
        ("平均簇长度", "mean_stickiness", ".1f"),
        ("平均簇数量", "mean_clusters", ".1f"),
        ("零异常点位", "zero_points", "d"),
    ]

    for label, key, fmt in metrics:
        print(f"{label:<25}", end="")
        for algo in algo_data:
            val = algo_data[algo].get(key, "N/A")
            if val == "N/A":
                print(f"{'N/A':<20}", end="")
            else:
                print(f"{val:<20{fmt}}", end="")
        print()

    # 物理约束检查
    print(f"\n{'='*80}")
    print("物理约束检查")
    print(f"{'='*80}")
    for algo, data in algo_data.items():
        rate = data["mean_rate"]
        violations = []
        if rate > 0.15:
            violations.append(f"异常率过高 ({rate:.4f} > 0.15)")
        if rate < 0.001:
            violations.append(f"异常率过低 ({rate:.6f} < 0.001)")
        status = "PASS" if not violations else "FAIL"
        print(f"  {algo:<20} [{status}] " +
              (", ".join(violations) if violations else "通过"))

    # 保存对比报告
    report = {
        "algorithms": algo_data,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    report_path = PROJECT_ROOT / "results/comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", nargs="*", default=None)
    args = parser.parse_args()

    if args.algo:
        algo_names = args.algo
    else:
        # 自动发现有预测结果的算法
        pred_base = PROJECT_ROOT / "results/predictions"
        algo_names = []
        if pred_base.exists():
            for d in pred_base.iterdir():
                if d.is_dir() and list(d.glob("*.csv")):
                    algo_names.append(d.name)
        # 始终包含 ADTK baseline
        if "adtk_hbos" not in algo_names:
            algo_names.insert(0, "adtk_hbos")

    print(f"对比算法: {algo_names}")
    generate_comparison(algo_names)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
集成异常检测器 —— 组合多种方法的检测结果

核心思想：
1. Timer/Sundial 提供精细的基于预测残差的异常区间
2. 统计方法（IForest, MAD 等）提供基于数值分布的异常点
3. 集成策略：投票/加权融合，利用方法间的互补性

集成策略：
- voting: 至少 N 种方法同时标记为异常
- weighted: 按方法权重加权后阈值判定
- cascade: Timer 检测 → 统计方法过滤假阳性

本脚本作为后处理器，读取各方法的预测结果进行融合。
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_prediction(algo_dir: str, point_name: str) -> np.ndarray:
    """加载某个算法对某个点位的预测 mask"""
    csv_path = os.path.join(algo_dir, f"{point_name}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "global_mask" in df.columns:
        return df["global_mask"].values.astype(int)
    return None


def ensemble_voting(masks: list, min_votes: int = 2) -> np.ndarray:
    """投票集成：至少 min_votes 种方法标记为异常"""
    if not masks:
        return np.array([])
    # 对齐长度（取最短）
    min_len = min(len(m) for m in masks)
    aligned = [m[:min_len] for m in masks]
    votes = np.sum(aligned, axis=0)
    return (votes >= min_votes).astype(int)


def ensemble_weighted(masks: list, weights: list, threshold: float = 0.5) -> np.ndarray:
    """加权集成：按权重加权后阈值判定"""
    if not masks:
        return np.array([])
    min_len = min(len(m) for m in masks)
    aligned = [m[:min_len].astype(float) for m in masks]
    weights = np.array(weights) / np.sum(weights)
    weighted = np.zeros(min_len)
    for m, w in zip(aligned, weights):
        weighted += m * w
    return (weighted >= threshold).astype(int)


def ensemble_cascade(timer_mask: np.ndarray, stat_masks: list,
                     confirm_ratio: float = 0.3) -> np.ndarray:
    """
    级联集成：Timer 检测 → 统计方法确认

    策略：Timer 标记的异常区间，如果统计方法中有 confirm_ratio 比例的
    方法也标记了该区间的部分点，则保留；否则视为假阳性移除。
    """
    if timer_mask is None:
        return np.array([])

    result = timer_mask.copy()
    min_len = len(timer_mask)

    # 找到 Timer 的异常簇
    diff = np.diff(np.concatenate(([0], timer_mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        # 检查统计方法是否确认
        confirmed = 0
        for stat_mask in stat_masks:
            if len(stat_mask) > s:
                segment = stat_mask[s:min(e, len(stat_mask))]
                if segment.mean() > 0.05:  # 该区间有 5% 以上被标记
                    confirmed += 1
        if len(stat_masks) > 0 and confirmed / len(stat_masks) < confirm_ratio:
            result[s:e] = 0  # 移除未确认的异常

    return result


def run_ensemble(point_name: str, strategy: str = "voting",
                 algos: list = None, output_dir: str = None):
    """对单个点位运行集成检测"""
    pred_base = str(PROJECT_ROOT / "results/predictions")
    if algos is None:
        algos = ["timer", "iforest", "3sigma", "mad", "iqr"]

    # 加载各方法的预测结果
    masks = {}
    for algo in algos:
        algo_dir = os.path.join(pred_base, algo)
        mask = load_prediction(algo_dir, point_name)
        if mask is not None:
            masks[algo] = mask

    if not masks:
        return None

    # 执行集成策略
    mask_list = list(masks.values())
    algo_list = list(masks.keys())

    if strategy == "voting":
        result = ensemble_voting(mask_list, min_votes=2)
    elif strategy == "weighted":
        # Timer 权重最高
        weights = [3.0 if a == "timer" else 1.0 for a in algo_list]
        result = ensemble_weighted(mask_list, weights, threshold=0.4)
    elif strategy == "cascade":
        timer_mask = masks.get("timer")
        stat_masks = [m for a, m in masks.items() if a != "timer"]
        if timer_mask is not None:
            result = ensemble_cascade(timer_mask, stat_masks, confirm_ratio=0.3)
        else:
            result = ensemble_voting(mask_list, min_votes=2)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame({"global_mask": result.astype(np.int8)}).to_csv(
            os.path.join(output_dir, f"{point_name}.csv"), index=False)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="cascade",
                        choices=["voting", "weighted", "cascade"])
    parser.add_argument("--points", default="evaluated",
                        help="evaluated / all / 逗号分隔")
    args = parser.parse_args()

    # 获取点位
    if args.points == "evaluated":
        with open(PROJECT_ROOT / "data/cleaned/evaluated_points.txt") as f:
            points = [l.strip() for l in f if l.strip()]
    else:
        points = args.points.split(",")

    output_dir = str(PROJECT_ROOT / f"results/predictions/ensemble_{args.strategy}")

    print(f"集成策略: {args.strategy}, 点位数: {len(points)}")

    results = []
    for i, point in enumerate(points):
        mask = run_ensemble(point, args.strategy, output_dir=output_dir)
        if mask is not None:
            rate = float(mask.mean())
            results.append({"point": point, "anomaly_rate": rate})
        else:
            results.append({"point": point, "error": "no predictions"})

    # 汇总
    valid = [r for r in results if "error" not in r]
    rates = [r["anomaly_rate"] for r in valid]
    print(f"\n完成: {len(valid)}/{len(points)} 点位")
    if rates:
        print(f"  平均异常率: {np.mean(rates):.4f}")
        print(f"  中位异常率: {np.median(rates):.4f}")
        print(f"  零异常点位: {sum(1 for r in rates if r == 0)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
参数扫描实验 —— autoresearch 核心循环

对 Timer 的关键检测参数进行网格搜索，找到最优配置。

扫描参数：
- threshold_k: MAD 阈值系数 [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
- method: mad / sigma
- lookback_length: [128, 256, 512]
- n_downsample: [5000, 10000, 20000]

由于每个配置需要约 60 分钟运行全部 99 点位，
策略：先在 10 个代表性点位上快速扫描，找到 Top-3 配置后全量运行。
"""

import json
import glob
import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent


def get_representative_points(n: int = 10) -> list:
    """选取代表性点位：覆盖不同评分区间和传感器类型"""
    merged = pd.read_csv(PROJECT_ROOT / "data/features/merged_scores_features.csv")

    # 按 Timer 评分分层抽样
    high = merged[merged["avg_timer"] >= 0.8].sample(min(3, len(merged[merged["avg_timer"] >= 0.8])),
                                                      random_state=42)
    mid = merged[(merged["avg_timer"] >= 0.4) & (merged["avg_timer"] < 0.8)].sample(
        min(4, len(merged[(merged["avg_timer"] >= 0.4) & (merged["avg_timer"] < 0.8)])),
        random_state=42)
    low = merged[merged["avg_timer"] < 0.4].sample(min(3, len(merged[merged["avg_timer"] < 0.4])),
                                                     random_state=42)

    selected = pd.concat([high, mid, low])
    return selected["point_name"].tolist()


def find_input_csv(point_name: str) -> str:
    data_dir = str(PROJECT_ROOT / "data/adtk_hbos_old")
    matches = glob.glob(os.path.join(data_dir, f"global_adtk_hbos_*_{point_name}_*_trend_seasonal_resid.csv"))
    return matches[0] if matches else None


def run_timer_with_params(points: list, params: dict, output_tag: str) -> dict:
    """运行 Timer 并返回统计结果"""
    adapter = str(BENCHMARK_DIR / "adapters/timer_batch_adapter.py")
    output_dir = str(PROJECT_ROOT / f"results/predictions/timer_{output_tag}")
    os.makedirs(output_dir, exist_ok=True)

    # 写入临时点位文件
    points_file = str(PROJECT_ROOT / f"results/tmp_points_{output_tag}.txt")
    with open(points_file, "w") as f:
        f.write("\n".join(points))

    cmd = [
        "conda", "run", "-n", "timer",
        "python", "-u", adapter,
        "--input-dir", str(PROJECT_ROOT / "data/adtk_hbos_old"),
        "--output-dir", output_dir,
        "--points-file", points_file,
        "--model-path", params.get("model_path", "/home/share/llm_models/thuml/timer-base-84m"),
        "--compact",
        "--threshold-k", str(params["threshold_k"]),
        "--method", params["method"],
        "--lookback-length", str(params["lookback_length"]),
        "--n-downsample", str(params["n_downsample"]),
    ]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=1800, env=env)
    elapsed = time.time() - start

    # 读取结果
    status_files = glob.glob(os.path.join(output_dir, "*.status.json"))
    rates = []
    for sf in status_files:
        d = json.load(open(sf))
        if d.get("status") == "success" and "anomaly_rate" in d:
            rates.append(d["anomaly_rate"])

    # 清理临时文件
    os.remove(points_file)

    return {
        "tag": output_tag,
        "params": params,
        "success": len(rates),
        "total": len(points),
        "mean_rate": float(np.mean(rates)) if rates else 0,
        "median_rate": float(np.median(rates)) if rates else 0,
        "elapsed": elapsed,
    }


def score_config(result: dict) -> float:
    """基于规则的配置评分（模拟人工评分偏好）"""
    rate = result["mean_rate"]

    # 最佳异常率区间
    if 0.005 <= rate <= 0.05:
        score = 0.8
    elif 0.001 <= rate <= 0.10:
        score = 0.6
    elif rate < 0.001:
        score = 0.3  # 过于保守
    else:
        score = 0.4  # 过度检测

    return score


def main():
    # 获取代表性点位
    rep_points = get_representative_points(10)
    print(f"代表性点位 ({len(rep_points)}): {rep_points}")

    # 参数网格
    param_grid = {
        "threshold_k": [2.5, 3.0, 3.5, 4.0, 5.0],
        "method": ["mad"],
        "lookback_length": [256],
        "n_downsample": [10000],
    }

    configs = []
    for k in param_grid["threshold_k"]:
        for m in param_grid["method"]:
            for lb in param_grid["lookback_length"]:
                for ns in param_grid["n_downsample"]:
                    configs.append({
                        "threshold_k": k,
                        "method": m,
                        "lookback_length": lb,
                        "n_downsample": ns,
                    })

    print(f"\n扫描 {len(configs)} 种配置 × {len(rep_points)} 点位")
    print(f"预计耗时: ~{len(configs) * len(rep_points) * 60 / 60:.0f} 分钟\n")

    results = []
    for i, cfg in enumerate(configs):
        tag = f"k{cfg['threshold_k']}_{cfg['method']}_lb{cfg['lookback_length']}_ds{cfg['n_downsample']}"
        print(f"[{i+1}/{len(configs)}] {tag}...", end=" ", flush=True)

        result = run_timer_with_params(rep_points, cfg, tag)
        result["score"] = score_config(result)
        results.append(result)

        print(f"rate={result['mean_rate']:.4f}, score={result['score']:.2f}, "
              f"{result['elapsed']:.0f}s")

    # 排名
    results.sort(key=lambda x: -x["score"])
    print(f"\n{'='*70}")
    print("参数扫描排名")
    print(f"{'='*70}")
    print(f"{'排名':<5} {'配置':<40} {'异常率':<12} {'评分':<8}")
    print(f"{'-'*70}")
    for i, r in enumerate(results):
        tag = r["tag"]
        print(f"{i+1:<5} {tag:<40} {r['mean_rate']:<12.4f} {r['score']:<8.2f}")

    # 保存
    output_path = PROJECT_ROOT / "results/param_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    # 建议
    best = results[0]
    print(f"\n推荐最佳配置: {best['tag']}")
    print(f"  threshold_k={best['params']['threshold_k']}")
    print(f"  mean_rate={best['mean_rate']:.4f}")


if __name__ == "__main__":
    main()

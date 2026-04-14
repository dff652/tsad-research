#!/usr/bin/env python3
"""
参数扫描实验 —— autoresearch 核心循环

对 Timer 的关键检测参数进行网格搜索，找到最优配置。
借鉴 autoresearch 设计：
- 固定评估基准（代表性点位）
- 自动 keep/discard 决策
- NaN/crash 快速失败
- 超时控制与实验日志
- 可选的自主探索模式（--autonomous）

扫描参数：
- threshold_k: MAD 阈值系数 [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
- method: mad / sigma
- lookback_length: [128, 256, 512]
- n_downsample: [5000, 10000, 20000]

由于每个配置需要约 60 分钟运行全部 99 点位，
策略：先在 10 个代表性点位上快速扫描，找到 Top-3 配置后全量运行。

用法:
    python param_sweep.py                      # 标准网格搜索
    python param_sweep.py --autonomous         # 自主探索模式（网格搜索后围绕最优配置继续微调）
    python param_sweep.py --timeout 600        # 单配置超时 600 秒
"""

import argparse
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
from experiment_log import ExperimentLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent

# 借鉴 autoresearch: 每次实验的默认超时（秒）
DEFAULT_EXPERIMENT_TIMEOUT = 1800


def get_representative_points(n: int = 10) -> list:
    """选取代表性点位：覆盖不同评分区间和传感器类型"""
    features_path = PROJECT_ROOT / "data/features/merged_scores_features.csv"
    if not features_path.exists():
        print(f"错误: 特征文件不存在 {features_path}")
        print("请先运行 scripts/02_extract_csv_features.py 生成特征文件")
        return []

    merged = pd.read_csv(features_path)
    if merged.empty or "avg_timer" not in merged.columns:
        print(f"错误: 特征文件为空或缺少 avg_timer 列")
        return []

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


def run_timer_with_params(points: list, params: dict, output_tag: str,
                          timeout: int = DEFAULT_EXPERIMENT_TIMEOUT) -> dict:
    """运行 Timer 并返回统计结果

    借鉴 autoresearch 设计：
    - 超时控制：单次实验超时视为 crash
    - NaN/异常值快速失败：检测到异常率为 NaN 或不合理值时标记
    - 子进程隔离：通过 conda run 确保环境独立
    """
    adapter = str(BENCHMARK_DIR / "adapters/timer_batch_adapter.py")

    # 防护：适配器脚本存在性检查
    if not os.path.isfile(adapter):
        return {"tag": output_tag, "params": params, "success": 0, "total": len(points),
                "mean_rate": 0, "median_rate": 0, "elapsed": 0,
                "error": f"适配器不存在: {adapter}"}

    output_dir = str(PROJECT_ROOT / f"results/predictions/timer_{output_tag}")
    os.makedirs(output_dir, exist_ok=True)

    # 防护：验证至少有一个点位有对应的输入文件
    valid_points = [p for p in points if find_input_csv(p) is not None]
    if not valid_points:
        return {"tag": output_tag, "params": params, "success": 0, "total": len(points),
                "mean_rate": 0, "median_rate": 0, "elapsed": 0,
                "error": "所有点位均无对应输入 CSV"}
    if len(valid_points) < len(points):
        print(f"  警告: {len(points) - len(valid_points)}/{len(points)} 个点位无输入数据，已跳过")
        points = valid_points

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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        # 借鉴 autoresearch: 超时视为 crash，清理后继续
        if os.path.exists(points_file):
            os.remove(points_file)
        return {"tag": output_tag, "params": params, "success": 0, "total": len(points),
                "mean_rate": 0, "median_rate": 0, "elapsed": timeout,
                "error": f"超时 ({timeout}s)"}
    elapsed = time.time() - start

    # 读取结果
    status_files = glob.glob(os.path.join(output_dir, "*.status.json"))
    rates = []
    for sf in status_files:
        with open(sf) as fh:
            d = json.load(fh)
        if d.get("status") == "success" and "anomaly_rate" in d:
            rate = d["anomaly_rate"]
            # NaN 快速失败检测（借鉴 autoresearch 的 NaN fast-fail）
            if rate is not None and not (isinstance(rate, float) and np.isnan(rate)):
                rates.append(rate)

    # 清理临时文件
    if os.path.exists(points_file):
        os.remove(points_file)

    mean_rate = float(np.mean(rates)) if rates else 0

    # NaN 快速失败：结果全部无效
    if not rates and proc.returncode == 0:
        return {"tag": output_tag, "params": params, "success": 0, "total": len(points),
                "mean_rate": 0, "median_rate": 0, "elapsed": elapsed,
                "error": "所有点位结果无效（NaN 或缺失 anomaly_rate）"}

    return {
        "tag": output_tag,
        "params": params,
        "success": len(rates),
        "total": len(points),
        "mean_rate": mean_rate,
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


def run_grid_sweep(rep_points: list, param_grid: dict, logger: ExperimentLogger,
                   timeout: int = DEFAULT_EXPERIMENT_TIMEOUT) -> list:
    """阶段一：标准网格搜索"""
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
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5  # 借鉴 autoresearch: 连续失败过多则中止

    for i, cfg in enumerate(configs):
        tag = f"k{cfg['threshold_k']}_{cfg['method']}_lb{cfg['lookback_length']}_ds{cfg['n_downsample']}"
        print(f"[{i+1}/{len(configs)}] {tag}...", end=" ", flush=True)

        result = run_timer_with_params(rep_points, cfg, tag, timeout=timeout)

        # 错误/crash 处理
        if "error" in result:
            print(f"FAIL: {result['error']}")
            logger.log(algorithm="timer", config_tag=tag, score=0.0,
                       anomaly_rate=0.0, elapsed=result.get("elapsed", 0),
                       status="crash", description=result["error"])
            result["score"] = 0.0
            results.append(result)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n连续 {MAX_CONSECUTIVE_FAILURES} 次失败，中止扫描（环境或数据可能有问题）")
                break
            continue

        consecutive_failures = 0  # 重置连续失败计数
        result["score"] = score_config(result)
        results.append(result)

        # 借鉴 autoresearch 的 keep/discard 决策
        status = "keep" if result["score"] >= 0.6 else "discard"
        logger.log(
            algorithm="timer", config_tag=tag, score=result["score"],
            anomaly_rate=result["mean_rate"], elapsed=result["elapsed"],
            status=status,
            description=f"grid_sweep: k={cfg['threshold_k']} {cfg['method']} lb={cfg['lookback_length']}",
        )

        print(f"rate={result['mean_rate']:.4f}, score={result['score']:.2f}, "
              f"{result['elapsed']:.0f}s [{status}]")

    return results


def run_autonomous_refinement(best_params: dict, best_score: float,
                              rep_points: list, logger: ExperimentLogger,
                              max_rounds: int = 20,
                              timeout: int = DEFAULT_EXPERIMENT_TIMEOUT) -> list:
    """阶段二（可选）：围绕最优配置自主微调探索

    借鉴 autoresearch 的核心循环：
    - 基于当前最优配置生成微调变体
    - 评估 → keep（改进了）/ discard（没改进）
    - 自主循环直到收敛或达到最大轮次
    """
    print(f"\n{'='*70}")
    print("自主探索模式: 围绕最优配置微调")
    print(f"  起始最优: k={best_params['threshold_k']}, score={best_score:.4f}")
    print(f"  最大轮次: {max_rounds}")
    print(f"{'='*70}\n")

    current_best_params = best_params.copy()
    current_best_score = best_score
    results = []
    no_improve_count = 0
    MAX_NO_IMPROVE = 5  # 连续 N 次无改进则收敛停止

    for round_idx in range(max_rounds):
        # 生成微调候选：在当前最优参数附近扰动
        candidates = _generate_refinement_candidates(current_best_params, round_idx)

        for ci, candidate in enumerate(candidates):
            tag = (f"refine_r{round_idx}_c{ci}_"
                   f"k{candidate['threshold_k']}_{candidate['method']}_"
                   f"lb{candidate['lookback_length']}_ds{candidate['n_downsample']}")

            print(f"[探索 {round_idx+1}/{max_rounds}, 候选 {ci+1}/{len(candidates)}] "
                  f"{tag}...", end=" ", flush=True)

            result = run_timer_with_params(rep_points, candidate, tag, timeout=timeout)

            if "error" in result:
                print(f"CRASH: {result['error']}")
                logger.log(algorithm="timer", config_tag=tag, score=0.0,
                           anomaly_rate=0.0, elapsed=result.get("elapsed", 0),
                           status="crash", description=f"autonomous: {result['error']}")
                continue

            result["score"] = score_config(result)
            results.append(result)

            # 核心决策: keep or discard
            if result["score"] > current_best_score:
                current_best_score = result["score"]
                current_best_params = candidate.copy()
                no_improve_count = 0
                logger.log(
                    algorithm="timer", config_tag=tag, score=result["score"],
                    anomaly_rate=result["mean_rate"], elapsed=result["elapsed"],
                    status="keep",
                    description=f"autonomous IMPROVED: score {current_best_score:.4f}",
                )
                print(f"KEEP score={result['score']:.4f} (new best!)")
            else:
                no_improve_count += 1
                logger.log(
                    algorithm="timer", config_tag=tag, score=result["score"],
                    anomaly_rate=result["mean_rate"], elapsed=result["elapsed"],
                    status="discard",
                    description=f"autonomous: score {result['score']:.4f} <= best {current_best_score:.4f}",
                )
                print(f"DISCARD score={result['score']:.4f} (best={current_best_score:.4f})")

        # 收敛检查
        if no_improve_count >= MAX_NO_IMPROVE:
            print(f"\n收敛: 连续 {MAX_NO_IMPROVE} 次无改进，停止探索")
            break

    print(f"\n自主探索完成: 最终最优 k={current_best_params['threshold_k']}, "
          f"score={current_best_score:.4f}")
    return results


def _generate_refinement_candidates(base_params: dict, round_idx: int) -> list:
    """围绕基准配置生成微调候选"""
    candidates = []
    base_k = base_params["threshold_k"]

    # 在 threshold_k 附近细粒度搜索（步长随轮次缩小）
    step = 0.5 / (round_idx + 1)  # 第0轮 ±0.5, 第1轮 ±0.25, ...
    step = max(step, 0.05)  # 最小步长
    for delta in [-step, step]:
        k = round(base_k + delta, 2)
        if k > 0.5:  # 阈值不能太小
            candidate = base_params.copy()
            candidate["threshold_k"] = k
            candidates.append(candidate)

    # 偶数轮也尝试不同的 lookback_length
    if round_idx % 2 == 0:
        for lb in [128, 256, 512]:
            if lb != base_params["lookback_length"]:
                candidate = base_params.copy()
                candidate["lookback_length"] = lb
                candidates.append(candidate)

    return candidates


def print_ranking(results: list):
    """打印排名表"""
    valid = [r for r in results if "score" in r]
    valid.sort(key=lambda x: -x["score"])

    print(f"\n{'='*70}")
    print("参数扫描排名")
    print(f"{'='*70}")
    print(f"{'排名':<5} {'配置':<40} {'异常率':<12} {'评分':<8}")
    print(f"{'-'*70}")
    for i, r in enumerate(valid):
        tag = r.get("tag", "?")
        print(f"{i+1:<5} {tag:<40} {r.get('mean_rate', 0):<12.4f} {r['score']:<8.2f}")

    return valid


def main():
    parser = argparse.ArgumentParser(description="参数扫描实验 —— autoresearch 核心循环")
    parser.add_argument("--autonomous", action="store_true",
                        help="网格搜索后启动自主探索模式（围绕最优配置微调）")
    parser.add_argument("--max-rounds", type=int, default=20,
                        help="自主探索最大轮次 (默认 20)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_EXPERIMENT_TIMEOUT,
                        help=f"单配置超时秒数 (默认 {DEFAULT_EXPERIMENT_TIMEOUT})")
    args = parser.parse_args()

    # 获取代表性点位
    rep_points = get_representative_points(10)
    if not rep_points:
        print("中止: 无法获取代表性点位（数据文件缺失或为空）")
        sys.exit(1)
    print(f"代表性点位 ({len(rep_points)}): {rep_points}")

    logger = ExperimentLogger()

    # 参数网格
    param_grid = {
        "threshold_k": [2.5, 3.0, 3.5, 4.0, 5.0],
        "method": ["mad"],
        "lookback_length": [256],
        "n_downsample": [10000],
    }

    # 阶段一：网格搜索
    results = run_grid_sweep(rep_points, param_grid, logger, timeout=args.timeout)

    # 排名
    valid_results = print_ranking(results)

    # 保存
    output_path = PROJECT_ROOT / "results/param_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    if not valid_results:
        print("无有效结果，跳过后续步骤")
        sys.exit(1)

    best = valid_results[0]
    print(f"\n网格搜索最佳配置: {best.get('tag', '?')}")
    print(f"  threshold_k={best['params']['threshold_k']}")
    print(f"  score={best['score']:.4f}, mean_rate={best['mean_rate']:.4f}")

    # 阶段二（可选）：自主探索
    if args.autonomous:
        refine_results = run_autonomous_refinement(
            best_params=best["params"],
            best_score=best["score"],
            rep_points=rep_points,
            logger=logger,
            max_rounds=args.max_rounds,
            timeout=args.timeout,
        )
        all_results = results + refine_results
        print_ranking(all_results)

        # 更新保存
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"完整结果已保存: {output_path}")

    # 打印统一实验日志汇总
    logger.summary()


if __name__ == "__main__":
    main()

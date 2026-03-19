#!/usr/bin/env python3
"""
Benchmark 主调度器 —— 子进程驱动的全自动评测流水线

核心设计原则（遵循研发红线）：
1. 每个算法通过 subprocess 在独立 conda 环境中调用
2. 仅通过文件 I/O 通信，杜绝模块间直接依赖
3. 单一算法失败不阻断整体流水线
4. 结果自动汇总到统一评估报告

用法：
    python runner.py                    # 运行所有算法
    python runner.py --algo adtk_hbos   # 运行指定算法
    python runner.py --eval-only        # 仅评估已有结果
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

# 项目根目录
BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent


def load_config() -> dict:
    config_path = BENCHMARK_DIR / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> str:
    """解析相对路径（相对于 benchmark 目录）"""
    if os.path.isabs(path):
        return path
    return str((BENCHMARK_DIR / path).resolve())


def get_point_names_from_precomputed(data_dir: str) -> list:
    """从预计算 CSV 文件名中提取点位名列表"""
    import re
    names = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        basename = os.path.basename(f)
        # global_adtk_hbos_m4_0.1_1200.0_{point_name}_20230718_trend_seasonal_resid.csv
        match = re.match(r"global_adtk_hbos_m4_0\.1_1200\.0_(.+)_20230718_trend_seasonal_resid\.csv", basename)
        if match:
            names.append(match.group(1))
    return names


def run_adapter_subprocess(algo_key: str, algo_cfg: dict, input_path: str,
                           output_path: str, point_name: str, timeout: int = 600) -> dict:
    """通过 subprocess 在指定 conda 环境中调用算法适配器"""
    conda_env = algo_cfg["conda_env"]
    adapter_script = resolve_path(algo_cfg["adapter"])

    # 确保所有路径为绝对路径
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    adapter_script = str(Path(adapter_script).resolve())

    cmd = [
        "conda", "run", "-n", conda_env,
        "--no-capture-output",
        "python", "-u", adapter_script,
        "--input", input_path,
        "--output", output_path,
        "--point-name", point_name,
    ]

    if algo_cfg.get("model_path"):
        cmd.extend(["--model-path", algo_cfg["model_path"]])

    # 隔离用户 site-packages，确保 conda 环境内部版本优先
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        elapsed = time.time() - start
        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "elapsed": elapsed,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed": timeout}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_precomputed_adapter(algo_key: str, algo_cfg: dict, config: dict) -> dict:
    """处理预计算结果的算法（如 ADTK+HBOS）—— 直接批量转换，无需逐个 subprocess"""
    import re as _re

    data_dir = resolve_path(config["data"]["adtk_hbos_precomputed"])
    pred_dir = resolve_path(config["output"]["predictions_dir"]) + f"/{algo_key}"
    os.makedirs(pred_dir, exist_ok=True)

    point_names = get_point_names_from_precomputed(data_dir)
    print(f"  [{algo_key}] 发现 {len(point_names)} 个预计算点位，批量转换中...")

    import pandas as pd
    success_count = 0
    error_count = 0

    for i, point in enumerate(point_names):
        try:
            # 找到原始 CSV
            pattern = os.path.join(data_dir, f"global_adtk_hbos_*_{point}_*_trend_seasonal_resid.csv")
            matches = glob.glob(pattern)
            if not matches:
                error_count += 1
                continue

            df = pd.read_csv(matches[0])

            # 找传感器值列
            value_col = None
            skip_cols = {"Time", "global_mask", "outlier_mask", "local_mask",
                         "global_mask_cluster", "local_mask_cluster"}
            for col in df.columns:
                if col not in skip_cols:
                    value_col = col
                    break

            result = pd.DataFrame({
                "Time": df["Time"],
                "value": df[value_col],
                "global_mask": df["global_mask"].astype(int),
                "outlier_mask": df.get("outlier_mask", df["global_mask"]).astype(int),
                "local_mask": df.get("local_mask", pd.Series(0, index=df.index)).astype(int),
            })

            output_path = os.path.join(pred_dir, f"{point}.csv")
            result.to_csv(output_path, index=False)
            success_count += 1

        except Exception as e:
            error_count += 1

        if (i + 1) % 100 == 0:
            print(f"  [{algo_key}] 进度: {i+1}/{len(point_names)}, 成功: {success_count}")

    print(f"  [{algo_key}] 完成: {success_count}/{len(point_names)} 成功, {error_count} 失败")
    return {"total": len(point_names), "success": success_count, "errors": error_count}


def run_algorithm(algo_key: str, algo_cfg: dict, config: dict) -> dict:
    """运行单个算法的完整推理流程"""
    print(f"\n{'='*60}")
    print(f"运行算法: {algo_cfg['name']} ({algo_key})")
    print(f"  环境: {algo_cfg['conda_env']}")
    print(f"  类型: {algo_cfg['type']}")
    print(f"{'='*60}")

    start = time.time()

    if algo_cfg.get("precomputed"):
        result = run_precomputed_adapter(algo_key, algo_cfg, config)
    else:
        # TODO: 非预计算算法的推理流程（需要读取原始数据并推理）
        print(f"  [{algo_key}] 非预计算算法推理暂未实现，跳过")
        result = {"total": 0, "success": 0, "skipped": True}

    elapsed = time.time() - start
    result["elapsed_total"] = elapsed
    return result


def run_evaluation(config: dict, algo_keys: list = None):
    """运行评估 —— 基于预提取特征文件，无需读取 46GB 原始 CSV"""
    sys.path.insert(0, str(BENCHMARK_DIR))
    from evaluator import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator(config)
    results_dir = resolve_path(config["output"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    algos = algo_keys or list(config["algorithms"].keys())
    for algo_key in algos:
        algo_name = config["algorithms"][algo_key]["name"]
        print(f"\n评估算法: {algo_name} ({algo_key})")
        eval_result = evaluator.evaluate_algorithm(algo_name)

        all_results[algo_key] = eval_result

        # 存储详细结果
        detail_path = os.path.join(results_dir, f"eval_detail_{algo_key}.json")
        with open(detail_path, "w") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
        print(f"  详细结果已保存: {detail_path}")

    # 汇总报告
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n汇总报告已保存: {summary_path}")

    # 打印摘要
    print_summary(all_results)
    return all_results


def print_summary(results: dict):
    """打印评估摘要表"""
    print(f"\n{'='*80}")
    print("Benchmark 评估摘要")
    print(f"{'='*80}")
    print(f"{'算法':<20} {'异常率(全量)':<15} {'异常率(评分)':<15} {'粘性':<10} {'跳变率':<10} {'物理合规':<10}")
    print(f"{'-'*80}")
    for algo_key, result in results.items():
        summary = result.get("summary", {})
        name = result.get("algorithm", algo_key)
        rate_all = summary.get("mean_anomaly_rate_all", 0)
        rate_eval = summary.get("mean_anomaly_rate_evaluated", 0)
        stick = summary.get("mean_stickiness", 0)
        jump = summary.get("mean_jump_ratio", 0)
        compliant = "PASS" if summary.get("physics_compliant") else "FAIL"
        print(f"{name:<20} {rate_all:<15.6f} {rate_eval:<15.6f} {stick:<10.2f} {jump:<10.4f} {compliant:<10}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="TSAD Benchmark Runner")
    parser.add_argument("--algo", nargs="*", default=None, help="指定运行的算法 (默认全部)")
    parser.add_argument("--eval-only", action="store_true", help="仅评估已有结果")
    parser.add_argument("--timeout", type=int, default=600, help="单点位超时秒数")
    args = parser.parse_args()

    config = load_config()

    if args.algo:
        algo_keys = args.algo
    else:
        algo_keys = list(config["algorithms"].keys())

    if not args.eval_only:
        # 运行推理
        for algo_key in algo_keys:
            if algo_key not in config["algorithms"]:
                print(f"未知算法: {algo_key}")
                continue
            run_algorithm(algo_key, config["algorithms"][algo_key], config)

    # 运行评估
    print("\n" + "="*60)
    print("开始评估...")
    print("="*60)
    run_evaluation(config, algo_keys)


if __name__ == "__main__":
    main()

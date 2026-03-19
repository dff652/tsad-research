#!/usr/bin/env python3
"""
批量推理脚本 —— 在指定点位集上运行 Timer/Sundial 推理

遵循红线：
- 通过 subprocess 调用适配器（松耦合）
- PYTHONNOUSERSITE=1 隔离环境
- 模型权重从 /home/share/llm_models 加载

用法:
    python batch_inference.py --algo timer --points evaluated   # 100 评分点位
    python batch_inference.py --algo sundial --points all       # 全量 436 点位
    python batch_inference.py --algo timer --points evaluated --resume  # 断点续跑
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent

ALGO_CONFIG = {
    "timer": {
        "conda_env": "timer",
        "model_path": "/home/share/llm_models/thuml/timer-base-84m",
        "adapter": str(BENCHMARK_DIR / "adapters/timer_adapter.py"),
    },
    "sundial": {
        "conda_env": "timer",  # 同一 conda 环境，不同模型权重
        "model_path": "/home/share/llm_models/thuml/sundial-base-128m",
        "adapter": str(BENCHMARK_DIR / "adapters/timer_adapter.py"),
    },
}


def get_point_names(mode: str) -> list:
    """获取点位列表"""
    if mode == "evaluated":
        points_file = PROJECT_ROOT / "data/cleaned/evaluated_points.txt"
        with open(points_file) as f:
            return [line.strip() for line in f if line.strip()]
    elif mode == "all":
        data_dir = PROJECT_ROOT / "data/adtk_hbos_old"
        names = []
        for f in sorted(data_dir.glob("*.csv")):
            m = re.match(
                r"global_adtk_hbos_m4_0\.1_1200\.0_(.+)_20230718_trend_seasonal_resid\.csv",
                f.name)
            if m:
                names.append(m.group(1))
        return names
    else:
        return mode.split(",")


def find_input_csv(point_name: str) -> str:
    """找到点位对应的原始 CSV"""
    data_dir = PROJECT_ROOT / "data/adtk_hbos_old"
    pattern = str(data_dir / f"global_adtk_hbos_*_{point_name}_*_trend_seasonal_resid.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return matches[0]


def run_single_point(algo: str, point_name: str, timeout: int = 300) -> dict:
    """运行单个点位的推理"""
    cfg = ALGO_CONFIG[algo]
    input_csv = find_input_csv(point_name)
    if not input_csv:
        return {"point": point_name, "status": "error", "error": "input CSV not found"}

    output_dir = PROJECT_ROOT / "results/predictions" / algo
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = str(output_dir / f"{point_name}.csv")

    cmd = [
        "conda", "run", "-n", cfg["conda_env"],
        "python", "-u", cfg["adapter"],
        "--input", input_csv,
        "--output", output_csv,
        "--point-name", point_name,
        "--model-path", cfg["model_path"],
    ]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            # 读取状态文件
            status_file = output_csv.replace(".csv", ".status.json")
            metrics = {}
            if os.path.exists(status_file):
                with open(status_file) as f:
                    metrics = json.load(f).get("metrics", {})
            return {
                "point": point_name, "status": "success",
                "elapsed": elapsed, **metrics,
            }
        else:
            return {
                "point": point_name, "status": "error",
                "elapsed": elapsed,
                "stderr": result.stderr[-300:] if result.stderr else "",
            }
    except subprocess.TimeoutExpired:
        return {"point": point_name, "status": "timeout", "elapsed": timeout}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=list(ALGO_CONFIG.keys()))
    parser.add_argument("--points", default="evaluated",
                        help="evaluated / all / 逗号分隔的点位名")
    parser.add_argument("--resume", action="store_true", help="跳过已有结果")
    parser.add_argument("--timeout", type=int, default=300, help="单点超时秒数")
    args = parser.parse_args()

    points = get_point_names(args.points)
    print(f"算法: {args.algo}, 点位数: {len(points)}, 模式: {args.points}")

    # 断点续跑：跳过已有结果
    if args.resume:
        output_dir = PROJECT_ROOT / "results/predictions" / args.algo
        existing = {f.stem for f in output_dir.glob("*.csv")} if output_dir.exists() else set()
        before = len(points)
        points = [p for p in points if p not in existing]
        print(f"  断点续跑: 跳过 {before - len(points)} 个已有结果，剩余 {len(points)}")

    # 进度跟踪
    log_path = PROJECT_ROOT / f"results/batch_{args.algo}_{args.points}.jsonl"
    results = []
    success = 0
    errors = 0

    for i, point in enumerate(points):
        print(f"\n[{i+1}/{len(points)}] {point}...", end=" ", flush=True)
        result = run_single_point(args.algo, point, args.timeout)
        results.append(result)

        if result["status"] == "success":
            success += 1
            rate = result.get("anomaly_rate", 0)
            elapsed = result.get("elapsed", 0)
            print(f"OK (rate={rate:.4f}, {elapsed:.1f}s)")
        else:
            errors += 1
            print(f"FAIL: {result.get('error', result.get('stderr', '')[:100])}")

        # 追加写入日志
        with open(log_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 最终汇总
    print(f"\n{'='*60}")
    print(f"批量推理完成: {args.algo}")
    print(f"  成功: {success}/{len(points)}")
    print(f"  失败: {errors}/{len(points)}")
    if results:
        success_results = [r for r in results if r["status"] == "success"]
        if success_results:
            rates = [r.get("anomaly_rate", 0) for r in success_results]
            times = [r.get("elapsed", 0) for r in success_results]
            print(f"  平均异常率: {sum(rates)/len(rates):.4f}")
            print(f"  平均耗时: {sum(times)/len(times):.1f}s")
            print(f"  总耗时: {sum(times)/60:.1f}min")
    print(f"日志: {log_path}")


if __name__ == "__main__":
    main()

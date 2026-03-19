#!/usr/bin/env python3
"""
Timer/Sundial 批量适配器 —— 一次加载模型，处理多个点位

相比 timer_adapter.py（每个点位一个 subprocess），本脚本：
1. 加载模型一次
2. 循环处理所有点位
3. 大幅减少模型加载开销

用法：
    PYTHONNOUSERSITE=1 conda run -n timer python timer_batch_adapter.py \
        --input-dir /path/to/adtk_hbos_old \
        --output-dir /path/to/predictions/timer \
        --points-file /path/to/evaluated_points.txt \
        --model-path /home/share/llm_models/thuml/timer-base-84m \
        --resume

环境要求: timer conda 环境
"""

import sys
import os
import time
import json
import glob
import re
import argparse
import numpy as np
import pandas as pd

# 业务平台路径
PLATFORM_INFERENCE = "/home/douff/ts/ts-iteration-loop/services/inference"
sys.path.insert(0, PLATFORM_INFERENCE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="ADTK 推理结果目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--points-file", default=None, help="点位列表文件（不指定则处理全量）")
    parser.add_argument("--model-path", required=True, help="模型权重路径")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-downsample", type=int, default=10000)
    parser.add_argument("--lookback-length", type=int, default=256)
    parser.add_argument("--threshold-k", type=float, default=3.5)
    parser.add_argument("--method", default="mad")
    parser.add_argument("--min-run", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compact", action="store_true",
                        help="紧凑输出：只保存 mask 列，不保存 value（节省磁盘）")
    return parser.parse_args()


def find_point_csv(input_dir: str, point_name: str) -> str:
    pattern = os.path.join(input_dir, f"global_adtk_hbos_*_{point_name}_*_trend_seasonal_resid.csv")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def get_all_points(input_dir: str) -> list:
    names = []
    for f in sorted(glob.glob(os.path.join(input_dir, "*.csv"))):
        m = re.match(
            r"global_adtk_hbos_m4_0\.1_1200\.0_(.+)_20230718_trend_seasonal_resid\.csv",
            os.path.basename(f))
        if m:
            names.append(m.group(1))
    return names


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取点位列表
    if args.points_file:
        with open(args.points_file) as f:
            points = [l.strip() for l in f if l.strip()]
    else:
        points = get_all_points(args.input_dir)

    # 断点续跑
    if args.resume:
        existing = {os.path.splitext(f)[0] for f in os.listdir(args.output_dir)
                     if f.endswith(".csv")}
        before = len(points)
        points = [p for p in points if p not in existing]
        print(f"断点续跑: 跳过 {before - len(points)} 个已有结果")

    print(f"待处理点位: {len(points)}, 模型: {args.model_path}")

    # 一次性加载模型
    from timer_detect import get_timer_pipeline, timer_detect
    pipeline = get_timer_pipeline(args.model_path, args.device)

    success = 0
    errors = 0
    total_time = 0

    for i, point in enumerate(points):
        csv_path = find_point_csv(args.input_dir, point)
        if not csv_path:
            print(f"[{i+1}/{len(points)}] {point} - CSV 未找到，跳过")
            errors += 1
            continue

        try:
            st = time.time()
            df = pd.read_csv(csv_path)

            # 找值列
            skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                    "global_mask_cluster", "local_mask_cluster"}
            value_col = [c for c in df.columns if c not in skip][0]

            # 运行 Timer 检测
            data_df = df[[value_col]].copy()
            global_mask, anomalies, _ = timer_detect(
                data=data_df,
                model_path=args.model_path,
                device=args.device,
                n_downsample=args.n_downsample,
                lookback_length=args.lookback_length,
                threshold_k=args.threshold_k,
                method=args.method,
                min_run=args.min_run,
            )

            # 保存结果
            output_path = os.path.join(args.output_dir, f"{point}.csv")
            if args.compact:
                # 紧凑模式：只保存 mask
                result = pd.DataFrame({
                    "global_mask": global_mask.astype(np.int8),
                })
            else:
                result = pd.DataFrame({
                    "Time": df["Time"],
                    "value": df[value_col],
                    "global_mask": global_mask.astype(int),
                    "outlier_mask": global_mask.astype(int),
                    "local_mask": np.zeros(len(df), dtype=int),
                })
            result.to_csv(output_path, index=False)

            # 保存状态
            elapsed = time.time() - st
            total_time += elapsed
            anomaly_rate = float(global_mask.mean())

            status_path = os.path.join(args.output_dir, f"{point}.status.json")
            with open(status_path, "w") as f:
                json.dump({
                    "status": "success",
                    "anomaly_rate": anomaly_rate,
                    "anomaly_count": int(global_mask.sum()),
                    "num_intervals": len(anomalies),
                    "total_rows": len(df),
                    "elapsed": elapsed,
                }, f, indent=2)

            success += 1
            print(f"[{i+1}/{len(points)}] {point} - "
                  f"rate={anomaly_rate:.4f}, intervals={len(anomalies)}, "
                  f"{elapsed:.1f}s")

        except Exception as e:
            errors += 1
            elapsed = time.time() - st
            print(f"[{i+1}/{len(points)}] {point} - ERROR: {e}")
            # 保存错误状态
            status_path = os.path.join(args.output_dir, f"{point}.status.json")
            with open(status_path, "w") as f:
                json.dump({"status": "error", "error": str(e)}, f)

    # 汇总
    print(f"\n{'='*60}")
    print(f"完成: {success}/{len(points)} 成功, {errors} 失败")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    if success > 0:
        print(f"平均每点: {total_time/success:.1f} 秒")


if __name__ == "__main__":
    main()

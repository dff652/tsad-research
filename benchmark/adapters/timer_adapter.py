#!/usr/bin/env python3
"""
Timer/Sundial 适配器 —— 在 timer conda 环境中运行

通过 subprocess 调用，输入原始时序 CSV，输出标准格式异常检测结果。

用法（由 benchmark runner 自动调用）：
    conda run -n timer python timer_adapter.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv \
        --point-name FI6101.PV \
        --model-path /home/share/llm_models/thuml/timer-base-84m

环境要求: timer conda 环境（含 torch, transformers）
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd

# 添加业务平台路径以复用 signal_processing 等工具
PLATFORM_INFERENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
sys.path.insert(0, PLATFORM_INFERENCE)


def parse_args():
    parser = argparse.ArgumentParser(description="Timer/Sundial Anomaly Detection Adapter")
    parser.add_argument("--input", required=True, help="输入 CSV（含 Time + 传感器值列）")
    parser.add_argument("--output", required=True, help="输出标准格式 CSV")
    parser.add_argument("--point-name", default="unknown", help="点位名称")
    parser.add_argument("--model-path", default="/home/share/llm_models/thuml/timer-base-84m",
                        help="模型权重路径")
    parser.add_argument("--device", default="cuda:0", help="GPU 设备")
    parser.add_argument("--n-downsample", type=int, default=10000, help="降采样点数")
    parser.add_argument("--lookback-length", type=int, default=256, help="滚动窗口长度")
    parser.add_argument("--threshold-k", type=float, default=3.5, help="异常阈值系数")
    parser.add_argument("--method", default="mad", choices=["mad", "sigma"], help="检测方法")
    parser.add_argument("--min-run", type=int, default=1, help="最小连续异常点数")
    return parser.parse_args()


def load_input(filepath: str) -> tuple:
    """加载输入 CSV，返回 (DataFrame, value_column_name)"""
    df = pd.read_csv(filepath)
    if "Time" not in df.columns and "time" in df.columns:
        df.rename(columns={"time": "Time"}, inplace=True)

    # 找到值列（非 Time、非 mask 列）
    skip = {"Time", "time", "global_mask", "outlier_mask", "local_mask",
            "global_mask_cluster", "local_mask_cluster"}
    value_col = None
    for col in df.columns:
        if col not in skip:
            value_col = col
            break

    if value_col is None:
        raise ValueError(f"无法识别值列: {df.columns.tolist()}")

    return df, value_col


def write_status(output_path: str, status: str, metrics: dict = None):
    status_path = output_path.replace(".csv", ".status.json")
    info = {
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics or {}
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(status_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    try:
        # 加载数据
        df, value_col = load_input(args.input)
        print(f"[Timer-Adapter] 点位: {args.point_name}, 数据量: {len(df)}, 值列: {value_col}")

        # 准备输入（timer_detect 需要的格式）
        data_df = df[[value_col]].copy()

        # 调用 Timer 检测
        from timer_detect import timer_detect

        global_mask, anomalies, position_index = timer_detect(
            data=data_df,
            model_path=args.model_path,
            device=args.device,
            n_downsample=args.n_downsample,
            lookback_length=args.lookback_length,
            threshold_k=args.threshold_k,
            method=args.method,
            min_run=args.min_run,
        )

        # 构造标准输出
        result = pd.DataFrame({
            "Time": df["Time"],
            "value": df[value_col],
            "global_mask": global_mask.astype(int),
            "outlier_mask": global_mask.astype(int),  # Timer 不区分 outlier/local
            "local_mask": np.zeros(len(df), dtype=int),
        })

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        result.to_csv(args.output, index=False)

        anomaly_rate = float(global_mask.mean())
        print(f"[Timer-Adapter] 完成: 异常率={anomaly_rate:.6f}, 异常区间={len(anomalies)}")

        write_status(args.output, "success", {
            "anomaly_rate": anomaly_rate,
            "anomaly_count": int(global_mask.sum()),
            "num_intervals": len(anomalies),
            "total_rows": len(df),
            "model_path": args.model_path,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        write_status(args.output, "error", {"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()

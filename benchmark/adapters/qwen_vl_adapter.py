#!/usr/bin/env python3
"""
Qwen-VL 适配器 —— 在 qwen_tune conda 环境中运行

Qwen-VL 是阿里的视觉语言模型，通过"看图"识别时序异常。
先将时序数据绘制为图片，再让 VLM 分析图中的异常区域。

用法：
    PYTHONNOUSERSITE=1 conda run -n qwen_tune python qwen_vl_adapter.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv \
        --point-name FI6101.PV \
        --model-path /home/share/llm_models/Qwen/Qwen3-VL-8B-Instruct
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd

# 业务平台路径
PLATFORM_INFERENCE = "/home/douff/ts/ts-iteration-loop/services/inference"
sys.path.insert(0, PLATFORM_INFERENCE)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL Anomaly Detection Adapter")
    parser.add_argument("--input", required=True, help="输入 CSV")
    parser.add_argument("--output", required=True, help="输出标准格式 CSV")
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--model-path", default="/home/share/llm_models/Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-downsample", type=int, default=1000, help="降采样点数（VL 图像分辨率限制）")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--compact", action="store_true", help="紧凑输出")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # 加载数据
        df = pd.read_csv(args.input)
        if "Time" not in df.columns and "time" in df.columns:
            df.rename(columns={"time": "Time"}, inplace=True)

        skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        value_col = [c for c in df.columns if c not in skip][0]
        print(f"[Qwen-VL-Adapter] 点位: {args.point_name}, 数据量: {len(df)}, 值列: {value_col}")

        # 调用 Qwen 检测
        from qwen_detect import qwen_detect

        data_df = df[[value_col]].copy()

        global_mask, anomalies, position_index = qwen_detect(
            data=data_df,
            model_path=args.model_path,
            device=args.device,
            n_downsample=args.n_downsample,
            max_new_tokens=args.max_new_tokens,
        )

        # 保存结果
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.compact:
            result = pd.DataFrame({"global_mask": global_mask.astype(np.int8)})
        else:
            result = pd.DataFrame({
                "Time": df["Time"],
                "value": df[value_col],
                "global_mask": global_mask.astype(int),
                "outlier_mask": global_mask.astype(int),
                "local_mask": np.zeros(len(df), dtype=int),
            })
        result.to_csv(args.output, index=False)

        anomaly_rate = float(global_mask.mean())
        print(f"[Qwen-VL-Adapter] 完成: rate={anomaly_rate:.6f}, intervals={len(anomalies)}")

        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({
                "status": "success",
                "anomaly_rate": anomaly_rate,
                "anomaly_count": int(global_mask.sum()),
                "num_intervals": len(anomalies),
                "total_rows": len(df),
                "model_path": args.model_path,
            }, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

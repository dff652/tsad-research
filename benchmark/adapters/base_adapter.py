"""
算法适配器基类 —— 所有算法适配器的标准接口定义

每个算法适配器必须实现:
1. 从标准 CSV 读取时序数据
2. 运行异常检测
3. 输出标准格式 CSV（Time, value, global_mask, outlier_mask, local_mask）

适配器作为独立脚本运行，通过 subprocess 在各自 conda 环境中调用。
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path


def parse_standard_args():
    """解析标准命令行参数"""
    parser = argparse.ArgumentParser(description="TSAD Algorithm Adapter")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", required=True, help="输出 CSV 文件路径")
    parser.add_argument("--point-name", default=None, help="点位名称")
    parser.add_argument("--model-path", default=None, help="模型权重路径")
    parser.add_argument("--params", default="{}", help="JSON 格式额外参数")
    return parser.parse_args()


def load_input_csv(filepath: str) -> pd.DataFrame:
    """加载标准输入 CSV"""
    df = pd.read_csv(filepath)
    # 兼容不同的列名格式
    if "Time" not in df.columns and "time" in df.columns:
        df.rename(columns={"time": "Time"}, inplace=True)
    return df


def save_output_csv(df: pd.DataFrame, filepath: str):
    """保存标准输出 CSV，确保包含必需列"""
    required = ["Time", "value", "global_mask"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"输出缺少必需列: {col}")
    # 确保 mask 列为整数
    for col in ["global_mask", "outlier_mask", "local_mask"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def write_status(output_path: str, status: str, metrics: dict = None):
    """写入状态文件，供 benchmark runner 读取"""
    status_path = output_path.replace(".csv", ".status.json")
    info = {
        "status": status,  # "success" / "error" / "timeout"
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics or {}
    }
    with open(status_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

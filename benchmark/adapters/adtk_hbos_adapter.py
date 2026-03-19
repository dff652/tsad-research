"""
ADTK+HBOS 适配器 —— 读取已有预计算推理结果并转换为标准格式

已有 436 个点位的推理 CSV 文件位于 data/adtk_hbos_old/，
文件命名格式: global_adtk_hbos_m4_0.1_1200.0_{point_name}_20230718_trend_seasonal_resid.csv
输入列: Time, {point_name}, global_mask, outlier_mask, local_mask, global_mask_cluster, local_mask_cluster

本适配器将其标准化为统一输出格式。
"""

import sys
import os
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from base_adapter import parse_standard_args, save_output_csv, write_status


def find_precomputed_csv(data_dir: str, point_name: str) -> str:
    """根据点位名查找预计算结果 CSV"""
    pattern = os.path.join(data_dir, f"global_adtk_hbos_*_{point_name}_*_trend_seasonal_resid.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"未找到点位 {point_name} 的预计算结果: {pattern}")
    return matches[0]


def convert_to_standard(input_path: str, point_name: str) -> pd.DataFrame:
    """将 ADTK+HBOS 原始输出转换为标准格式"""
    df = pd.read_csv(input_path)

    # 找到传感器值列（非 Time、非 mask 列）
    value_col = None
    for col in df.columns:
        if col not in ("Time", "global_mask", "outlier_mask", "local_mask",
                        "global_mask_cluster", "local_mask_cluster"):
            value_col = col
            break

    if value_col is None:
        raise ValueError(f"无法识别传感器值列: {df.columns.tolist()}")

    result = pd.DataFrame({
        "Time": df["Time"],
        "value": df[value_col],
        "global_mask": df["global_mask"].astype(int),
        "outlier_mask": df.get("outlier_mask", df["global_mask"]).astype(int),
        "local_mask": df.get("local_mask", pd.Series(0, index=df.index)).astype(int),
    })
    return result


def main():
    args = parse_standard_args()

    try:
        if args.input and os.path.isdir(args.input):
            # 输入是预计算目录，根据 point_name 查找
            csv_path = find_precomputed_csv(args.input, args.point_name)
        else:
            csv_path = args.input

        result = convert_to_standard(csv_path, args.point_name)
        save_output_csv(result, args.output)

        metrics = {
            "total_rows": len(result),
            "anomaly_count": int(result["global_mask"].sum()),
            "anomaly_rate": float(result["global_mask"].mean()),
        }
        write_status(args.output, "success", metrics)

    except Exception as e:
        write_status(args.output, "error", {"error": str(e)})
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
快速运行所有统计方法基线

在已评分的 100 个点位上运行 IForest/3Sigma/MAD/IQR，
建立统计方法基线对比。
"""

import glob
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "adapters"))

from statistical_adapter import (
    detect_iforest, detect_3sigma, detect_iqr, detect_mad,
    downsample_m4, merge_short_clusters, map_mask_to_original
)

METHODS = {
    "iforest": lambda v: detect_iforest(v, contamination=0.01),
    "3sigma": lambda v: detect_3sigma(v, k=3.0),
    "mad": lambda v: detect_mad(v, k=3.5),
    "iqr": lambda v: detect_iqr(v, k=1.5),
}


def find_csv(data_dir: str, point: str) -> str:
    pattern = os.path.join(data_dir, f"global_adtk_hbos_*_{point}_*_trend_seasonal_resid.csv")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def main():
    data_dir = str(PROJECT_ROOT / "data/adtk_hbos_old")
    points_file = str(PROJECT_ROOT / "data/cleaned/evaluated_points.txt")

    with open(points_file) as f:
        points = [l.strip() for l in f if l.strip()]

    print(f"运行统计基线: {len(points)} 个评分点位, {len(METHODS)} 种方法")
    n_downsample = 50000

    results = {method: [] for method in METHODS}
    total_start = time.time()

    for i, point in enumerate(points):
        csv_path = find_csv(data_dir, point)
        if not csv_path:
            continue

        try:
            df = pd.read_csv(csv_path)
            skip = {"Time", "time", "global_mask", "outlier_mask", "local_mask",
                    "global_mask_cluster", "local_mask_cluster"}
            value_col = [c for c in df.columns if c not in skip][0]
            values = df[value_col].values.astype(np.float64)

            # 处理 NaN
            nan_mask = np.isnan(values)
            if nan_mask.any():
                values[nan_mask] = np.nanmedian(values)

            original_length = len(values)

            # 降采样
            if original_length > n_downsample:
                ds_values, ds_indices = downsample_m4(values, n_downsample)
            else:
                ds_values = values
                ds_indices = np.arange(original_length)

            for method_name, detect_fn in METHODS.items():
                st = time.time()
                mask_ds = detect_fn(ds_values)
                mask_ds = merge_short_clusters(mask_ds, 5)

                if len(ds_values) < original_length:
                    mask = map_mask_to_original(mask_ds, ds_indices, original_length)
                else:
                    mask = mask_ds

                elapsed = time.time() - st
                rate = float(mask.mean())

                results[method_name].append({
                    "point": point,
                    "anomaly_rate": rate,
                    "anomaly_count": int(mask.sum()),
                    "total_rows": original_length,
                    "elapsed": elapsed,
                })

                # 保存预测结果
                out_dir = PROJECT_ROOT / "results/predictions" / method_name
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(
                    out_dir / f"{point}.csv", index=False)

        except Exception as e:
            for method_name in METHODS:
                results[method_name].append({
                    "point": point, "error": str(e)})

        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start
            print(f"  进度: {i+1}/{len(points)}, 耗时: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n完成: {total_elapsed:.1f}s")

    # 汇总
    print(f"\n{'='*70}")
    print(f"{'方法':<15} {'平均异常率':<15} {'中位异常率':<15} {'成功点位':<10}")
    print(f"{'-'*70}")
    for method, records in results.items():
        valid = [r for r in records if "error" not in r]
        rates = [r["anomaly_rate"] for r in valid]
        if rates:
            print(f"{method:<15} {np.mean(rates):<15.6f} {np.median(rates):<15.6f} {len(valid):<10}")

    # 保存汇总
    summary = {}
    for method, records in results.items():
        valid = [r for r in records if "error" not in r]
        rates = [r["anomaly_rate"] for r in valid]
        if rates:
            summary[method] = {
                "mean_rate": float(np.mean(rates)),
                "median_rate": float(np.median(rates)),
                "std_rate": float(np.std(rates)),
                "max_rate": float(np.max(rates)),
                "min_rate": float(np.min(rates)),
                "valid_points": len(valid),
            }

    summary_path = PROJECT_ROOT / "results/statistical_baselines_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()

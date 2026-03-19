#!/usr/bin/env python3
"""
OpenCV 边缘检测异常检测适配器

来自 ilabel/cv_detect/cvd.py 的方法：
1. 将时序绘制为图片
2. 用 Canny 边缘检测 + Hough 变换检测垂直线段
3. 垂直线段 = 时序中的跳变/阶跃点
4. 将检测到的图像坐标映射回时序索引

这是基于计算机视觉的异常检测范式，完全不同于统计/预测方法。
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageChops
import io


def downsample_m4(values, n_out):
    n = len(values)
    if n <= n_out:
        return values, np.arange(n)
    chunk_size = max(1, n // (n_out // 4))
    indices = []
    for i in range(0, n, chunk_size):
        chunk = values[i:i + chunk_size]
        if len(chunk) == 0:
            continue
        idx_base = i
        indices.extend([idx_base, idx_base + int(np.argmin(chunk)),
                        idx_base + int(np.argmax(chunk)), idx_base + len(chunk) - 1])
    indices = sorted(set(indices))
    return values[indices], np.array(indices)


def fig_to_pil(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def crop_whitespace(im):
    img = im.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img


def detect_vertical_edges(image_path: str, series: np.ndarray,
                           min_line_length: int = 300, max_gap: int = 5) -> list:
    """
    用 OpenCV Canny + Hough 检测图像中的垂直线段（跳变）

    返回：[(data_index, magnitude), ...] 跳变点列表
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                             threshold=250, minLineLength=min_line_length,
                             maxLineGap=max_gap)

    data_len = len(series)
    img_width = edges.shape[1]
    detections = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)

            # 仅保留近似垂直的线段（dx <= 10）
            if dx <= 10:
                # 将图像 x 坐标映射到数据索引
                x_center = (x1 + x2) / 2
                data_idx = int((x_center / img_width) * data_len)
                data_idx = max(0, min(data_idx, data_len - 1))

                # 检查该位置附近是否有明显跳变
                window = max(1, data_len // 200)
                left = max(0, data_idx - window)
                right = min(data_len, data_idx + window)
                local_range = np.ptp(series[left:right]) if right > left else 0
                global_range = np.ptp(series)

                if global_range > 0 and local_range / global_range > 0.1:
                    detections.append((data_idx, float(local_range)))

    # 去重（合并相近的检测）
    if not detections:
        return []
    detections.sort()
    merged = [detections[0]]
    merge_dist = max(1, data_len // 100)
    for idx, mag in detections[1:]:
        if idx - merged[-1][0] < merge_dist:
            if mag > merged[-1][1]:
                merged[-1] = (idx, mag)
        else:
            merged.append((idx, mag))

    return merged


def create_mask_from_detections(data_len: int, detections: list,
                                 expand: int = 50) -> np.ndarray:
    """从检测到的跳变点创建 mask（每个检测点向两侧扩展）"""
    mask = np.zeros(data_len, dtype=int)
    for idx, _ in detections:
        start = max(0, idx - expand)
        end = min(data_len, idx + expand)
        mask[start:end] = 1
    return mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--n-downsample", type=int, default=10000)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        df = pd.read_csv(args.input)
        skip = {"Time", "time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        if "Time" not in df.columns and "time" in df.columns:
            df.rename(columns={"time": "Time"}, inplace=True)
        value_col = [c for c in df.columns if c not in skip][0]
        values = df[value_col].values.astype(np.float64)
        nan_mask = np.isnan(values)
        if nan_mask.any():
            values[nan_mask] = np.nanmedian(values)

        original_length = len(values)
        print(f"[OpenCV-Edge] 点位: {args.point_name}, 数据量: {original_length}")

        # 降采样
        if original_length > args.n_downsample:
            ds_values, ds_indices = downsample_m4(values, args.n_downsample)
        else:
            ds_values, ds_indices = values, np.arange(original_length)

        # 绘制时序图
        st = time.time()
        mpl.rcParams['path.simplify'] = True
        mpl.rcParams['path.simplify_threshold'] = 0.5
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ds_values, linewidth=0.3)
        ax.axis('off')
        img_pil = fig_to_pil(fig)
        img_cropped = crop_whitespace(img_pil)

        # 保存临时图片供 OpenCV 使用
        tmp_img = args.output.replace(".csv", "_tmp.png")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        img_cropped.save(tmp_img)

        # 检测垂直线段
        detections = []
        for min_len in (200, 300, 360):
            detections = detect_vertical_edges(tmp_img, ds_values, min_len, 5)
            if detections:
                break

        # 创建降采样 mask
        expand = max(10, len(ds_values) // 200)
        mask_ds = create_mask_from_detections(len(ds_values), detections, expand)

        # 映射回原始长度
        if len(ds_values) < original_length:
            mask = np.zeros(original_length, dtype=int)
            for i, idx in enumerate(ds_indices):
                if mask_ds[i] == 1:
                    mask[idx] = 1
            # 填充间隙
            chunk_size = max(1, original_length // len(ds_indices))
            diff_m = np.diff(np.concatenate(([0], mask, [0])))
            starts = np.where(diff_m == 1)[0]
            ends = np.where(diff_m == -1)[0]
            for si in range(len(starts) - 1):
                if starts[si + 1] - ends[si] <= chunk_size * 2:
                    mask[ends[si]:starts[si + 1]] = 1
        else:
            mask = mask_ds

        elapsed = time.time() - st
        rate = float(mask.mean())
        print(f"[OpenCV-Edge] 完成: rate={rate:.4f}, detections={len(detections)}, {elapsed:.2f}s")

        # 保存
        if args.compact:
            pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(args.output, index=False)
        else:
            pd.DataFrame({"Time": df["Time"], "value": df[value_col],
                          "global_mask": mask.astype(int)}).to_csv(args.output, index=False)

        # 清理临时图片
        os.remove(tmp_img)

        with open(args.output.replace(".csv", ".status.json"), "w") as f:
            json.dump({"status": "success", "anomaly_rate": rate,
                       "anomaly_count": int(mask.sum()), "num_detections": len(detections),
                       "total_rows": original_length, "elapsed": elapsed}, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output.replace(".csv", ".status.json"), "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
构建 Qwen3-VL Grounding 训练数据集

将时序异常检测转化为图像目标检测任务：
1. 从 ADTK+HBOS 的 CSV 读取时序数据和异常标注
2. 将时序绘制为图片（与 cvd.py 保持一致的绘图风格）
3. 将异常索引区间转换为图像上的 bbox_2d 坐标
4. 输出 Qwen3-VL grounding 格式的训练数据

Qwen3-VL 坐标系：相对坐标 0-1000
bbox_2d: [x1, y1, x2, y2]

数据集划分规则：
- 训练集：337 个非评分点位（data/cleaned/train_points.txt）
- 测试集：99 个有评分的点位（data/cleaned/evaluated_points.txt）—— 仅评估
"""

import os
import sys
import json
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 图像参数（与 cvd.py 保持一致）
FIG_WIDTH = 20      # 英寸
FIG_HEIGHT = 4      # 英寸
FIG_DPI = 200       # 分辨率
LINE_WIDTH = 0.5    # 线宽
IMG_WIDTH = FIG_WIDTH * FIG_DPI   # 4000 px
IMG_HEIGHT = FIG_HEIGHT * FIG_DPI  # 800 px


def downsample_m4(values: np.ndarray, n_out: int) -> tuple:
    """M4 降采样"""
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
        indices.extend([
            idx_base,
            idx_base + int(np.argmin(chunk)),
            idx_base + int(np.argmax(chunk)),
            idx_base + len(chunk) - 1,
        ])
    indices = sorted(set(indices))
    return values[indices], np.array(indices)


def plot_timeseries_to_image(values: np.ndarray, output_path: str,
                              fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT,
                              dpi=FIG_DPI, linewidth=LINE_WIDTH) -> dict:
    """
    将时序数据绘制为纯净的图片（无轴标签/刻度），返回坐标映射信息。

    使用 axis('off') + 无边距保存，使得数据区域几乎填满整张图片，
    这样 x_frac = index / data_len 直接映射到图像 x 坐标。
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(range(len(values)), values, linewidth=linewidth, color='#1f77b4')
    ax.set_xlim(0, len(values) - 1)

    y_min, y_max = ax.get_ylim()
    # 添加少量 y 边距
    y_margin = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    y_min_padded = y_min - y_margin
    y_max_padded = y_max + y_margin

    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(output_path)
    actual_w, actual_h = img.size

    return {
        "img_width": actual_w,
        "img_height": actual_h,
        "data_len": len(values),
        "x_range": (0, len(values) - 1),
        "y_range": (float(y_min_padded), float(y_max_padded)),
        # axis('off') + pad_inches=0 使得 axes 几乎填满图片
        "ax_bbox": (0.0, 0.0, 1.0, 1.0),
    }


def data_index_to_bbox_2d(start_idx: int, end_idx: int, data_len: int,
                           y_min: float, y_max: float, values: np.ndarray,
                           plot_info: dict) -> list:
    """
    将数据索引区间转换为 Qwen3-VL bbox_2d 坐标 [x1, y1, x2, y2]（0-1000）

    异常区间在时序图上对应的矩形：
    - x 方向：异常的起止索引映射到图像 x 坐标
    - y 方向：异常区间内数据的最小/最大值映射到图像 y 坐标
    """
    ax_x0, ax_y0, ax_w, ax_h = plot_info["ax_bbox"]

    # x 坐标映射：data_index → axes 相对位置 → 图像位置 → Qwen 0-1000
    x_frac_start = start_idx / max(1, data_len - 1)
    x_frac_end = end_idx / max(1, data_len - 1)

    # 在 axes 内的位置
    x1_ax = ax_x0 + x_frac_start * ax_w
    x2_ax = ax_x0 + x_frac_end * ax_w

    # y 坐标映射：异常区间内的值范围
    seg = values[start_idx:end_idx + 1]
    if len(seg) == 0:
        return None

    seg_min = float(np.min(seg))
    seg_max = float(np.max(seg))

    # y 轴在图像中是翻转的（图像上方 = y 大值）
    y_range = y_max - y_min
    if y_range < 1e-10:
        y_range = 1.0

    y_frac_top = 1.0 - (seg_max - y_min) / y_range  # 上边（小 y）
    y_frac_bottom = 1.0 - (seg_min - y_min) / y_range  # 下边（大 y）

    y1_ax = ax_y0 + y_frac_top * ax_h
    y2_ax = ax_y0 + y_frac_bottom * ax_h

    # 扩展 bbox 使其不太窄
    min_width = 0.005  # 最小宽度 0.5%
    if x2_ax - x1_ax < min_width:
        center = (x1_ax + x2_ax) / 2
        x1_ax = center - min_width / 2
        x2_ax = center + min_width / 2

    min_height = 0.05  # 最小高度 5%
    if y2_ax - y1_ax < min_height:
        center = (y1_ax + y2_ax) / 2
        y1_ax = center - min_height / 2
        y2_ax = center + min_height / 2

    # 转换为 Qwen3-VL 的 0-1000 坐标
    bbox = [
        int(np.clip(x1_ax * 1000, 0, 1000)),
        int(np.clip(y1_ax * 1000, 0, 1000)),
        int(np.clip(x2_ax * 1000, 0, 1000)),
        int(np.clip(y2_ax * 1000, 0, 1000)),
    ]

    return bbox


def extract_anomaly_clusters(mask: np.ndarray, min_length: int = 5) -> list:
    """从 mask 中提取异常簇"""
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    clusters = []
    for s, e in zip(starts, ends):
        if e - s >= min_length:
            clusters.append((int(s), int(e - 1)))
    return clusters


def classify_anomaly_type(values: np.ndarray, start: int, end: int) -> str:
    """根据异常区间的形态分类异常类型"""
    seg = values[start:end + 1]
    if len(seg) < 2:
        return "point_anomaly"

    # 计算前后对比
    pre_start = max(0, start - len(seg))
    pre = values[pre_start:start]
    post_end = min(len(values), end + 1 + len(seg))
    post = values[end + 1:post_end]

    seg_mean = np.mean(seg)
    seg_std = np.std(seg)
    pre_mean = np.mean(pre) if len(pre) > 0 else seg_mean
    post_mean = np.mean(post) if len(post) > 0 else seg_mean

    # 分类逻辑
    if end - start < 5:
        return "spike"  # 尖峰
    elif abs(seg_mean - pre_mean) > 3 * max(seg_std, 1e-8):
        if seg_mean > pre_mean:
            return "level_shift_up"  # 阶跃上升
        else:
            return "level_shift_down"  # 阶跃下降
    elif seg_std > 3 * max(np.std(pre) if len(pre) > 5 else 1, 1e-8):
        return "variance_change"  # 方差突变
    elif len(seg) > 50:
        return "trend_drift"  # 趋势漂移
    else:
        return "anomaly_segment"  # 一般异常段


def process_single_point(point_name: str, csv_path: str,
                          images_dir: str, n_downsample: int = 5000) -> dict:
    """处理单个点位：生成图像 + bbox 标注"""
    df = pd.read_csv(csv_path)

    # 找到值列和 mask 列
    skip = {"Time", "global_mask", "outlier_mask", "local_mask",
            "global_mask_cluster", "local_mask_cluster"}
    value_col = [c for c in df.columns if c not in skip][0]
    values = df[value_col].values.astype(np.float64)
    mask = df["global_mask"].values.astype(int)

    # 处理 NaN
    nan_mask = np.isnan(values)
    if nan_mask.any():
        values[nan_mask] = np.nanmedian(values)

    original_length = len(values)

    # 降采样（保持较高分辨率以保留异常特征）
    if original_length > n_downsample:
        ds_values, ds_indices = downsample_m4(values, n_downsample)
        # 同步降采样 mask
        ds_mask = mask[ds_indices]
    else:
        ds_values = values
        ds_indices = np.arange(original_length)
        ds_mask = mask

    # 绘制图像
    img_path = os.path.join(images_dir, f"{point_name}.png")
    plot_info = plot_timeseries_to_image(ds_values, img_path)

    # 提取异常簇
    clusters = extract_anomaly_clusters(ds_mask, min_length=3)

    # 转换为 bbox
    annotations = []
    y_min, y_max = plot_info["y_range"]

    for start, end in clusters:
        bbox = data_index_to_bbox_2d(
            start, end, len(ds_values), y_min, y_max, ds_values, plot_info)
        if bbox is None:
            continue

        atype = classify_anomaly_type(ds_values, start, end)
        annotations.append({
            "bbox_2d": bbox,
            "label": atype,
            "data_range": [int(start), int(end)],
            "original_range": [int(ds_indices[start]), int(ds_indices[min(end, len(ds_indices)-1)])],
        })

    return {
        "point_name": point_name,
        "image_path": img_path,
        "image_size": [plot_info["img_width"], plot_info["img_height"]],
        "data_length": original_length,
        "downsampled_length": len(ds_values),
        "num_anomalies": len(annotations),
        "annotations": annotations,
    }


def build_qwen_conversation(record: dict) -> dict:
    """构建 Qwen3-VL 格式的对话数据"""
    img_path = record["image_path"]
    annotations = record["annotations"]

    # 用户 prompt
    user_prompt = (
        'This is a time series plot from an industrial sensor. '
        'Locate every anomaly region in this image. '
        'For each anomaly, report bbox coordinates and anomaly type in JSON format like: '
        '{"bbox_2d": [x1, y1, x2, y2], "label": "anomaly_type"}'
    )

    # 标注答案
    answer = json.dumps(annotations, ensure_ascii=False)

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "metadata": {
            "point_name": record["point_name"],
            "num_anomalies": record["num_anomalies"],
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--n-downsample", type=int, default=5000)
    parser.add_argument("--max-points", type=int, default=0, help="0=全部")
    args = parser.parse_args()

    data_dir = str(PROJECT_ROOT / "data/adtk_hbos_old")
    images_dir = str(PROJECT_ROOT / "qwen/images")
    dataset_dir = str(PROJECT_ROOT / "qwen/dataset")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # 选择点位
    if args.split == "train":
        points_file = PROJECT_ROOT / "data/cleaned/train_points.txt"
    elif args.split == "test":
        points_file = PROJECT_ROOT / "data/cleaned/evaluated_points.txt"
    else:
        points_file = None

    if points_file:
        with open(points_file) as f:
            points = [l.strip() for l in f if l.strip()]
    else:
        points = []
        for f in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
            m = re.match(r"global_adtk_hbos_m4_0\.1_1200\.0_(.+)_20230718_trend_seasonal_resid\.csv",
                         os.path.basename(f))
            if m:
                points.append(m.group(1))

    if args.max_points > 0:
        points = points[:args.max_points]

    print(f"构建数据集: split={args.split}, {len(points)} 点位")

    records = []
    conversations = []
    success = 0

    for i, point in enumerate(points):
        matches = glob.glob(os.path.join(data_dir,
                    f"global_adtk_hbos_*_{point}_*_trend_seasonal_resid.csv"))
        if not matches:
            continue

        try:
            record = process_single_point(point, matches[0], images_dir, args.n_downsample)
            records.append(record)

            conv = build_qwen_conversation(record)
            conversations.append(conv)

            success += 1
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(points)}, 成功: {success}, "
                      f"最近: {point} ({record['num_anomalies']} anomalies)")
        except Exception as e:
            print(f"  ERROR {point}: {e}")

    # 保存
    records_path = os.path.join(dataset_dir, f"annotations_{args.split}.json")
    with open(records_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    conv_path = os.path.join(dataset_dir, f"conversations_{args.split}.jsonl")
    with open(conv_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # 统计
    total_anomalies = sum(r["num_anomalies"] for r in records)
    zero_anomaly = sum(1 for r in records if r["num_anomalies"] == 0)
    print(f"\n完成: {success}/{len(points)} 点位")
    print(f"  总异常区间: {total_anomalies}")
    print(f"  零异常点位: {zero_anomaly}")
    print(f"  标注文件: {records_path}")
    print(f"  对话文件: {conv_path}")


if __name__ == "__main__":
    main()

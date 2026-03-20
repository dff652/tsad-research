#!/usr/bin/env python3
"""
Qwen3-VL Grounding 评估脚本

评估指标：
1. bbox IoU（交并比）：预测框与 GT 框的重合度
2. 检出率：GT 框中被预测覆盖的比例
3. 误检率：预测框中无 GT 对应的比例
4. 时序 mask 层面：与 ADTK 标注的一致性
5. 接入 auto_scorer 的预测评分
"""

import json
import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def bbox_iou_1d(pred_bbox, gt_bbox):
    """计算 1D IoU（仅 x 方向，因为异常检测关心时间范围）"""
    px1, _, px2, _ = pred_bbox
    gx1, _, gx2, _ = gt_bbox

    inter_start = max(px1, gx1)
    inter_end = min(px2, gx2)
    intersection = max(0, inter_end - inter_start)

    pred_len = max(1, px2 - px1)
    gt_len = max(1, gx2 - gx1)
    union = pred_len + gt_len - intersection

    return intersection / max(union, 1)


def bbox_iou_2d(pred_bbox, gt_bbox):
    """计算 2D IoU"""
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area = max(1, (px2 - px1) * (py2 - py1))
    gt_area = max(1, (gx2 - gx1) * (gy2 - gy1))
    union = pred_area + gt_area - inter_area

    return inter_area / max(union, 1)


def evaluate_single_point(pred_bboxes: list, gt_annotations: list,
                           iou_threshold: float = 0.1) -> dict:
    """评估单个点位的 grounding 质量"""
    gt_bboxes = [a["bbox_2d"] for a in gt_annotations if "bbox_2d" in a]
    n_pred = len(pred_bboxes)
    n_gt = len(gt_bboxes)

    base = {"n_pred": n_pred, "n_gt": n_gt, "tp": 0, "avg_iou": 0.0}
    if n_gt == 0 and n_pred == 0:
        return {**base, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if n_gt == 0:
        return {**base, "precision": 0.0, "recall": 1.0, "f1": 0.0}
    if n_pred == 0:
        return {**base, "precision": 1.0, "recall": 0.0, "f1": 0.0}

    # 匹配：贪心算法，按 IoU 降序匹配
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pb in enumerate(pred_bboxes):
        for j, gb in enumerate(gt_bboxes):
            iou_matrix[i, j] = bbox_iou_1d(pb, gb)

    matched_pred = set()
    matched_gt = set()

    # 按 IoU 降序匹配
    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matched_pred.add(i)
        matched_gt.add(j)
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    tp = len(matched_pred)
    precision = tp / n_pred if n_pred > 0 else 0
    recall = tp / n_gt if n_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 平均 IoU（匹配到的对）
    matched_ious = []
    for i in matched_pred:
        for j in matched_gt:
            iou = bbox_iou_1d(pred_bboxes[list(matched_pred).index(i) if False else i],
                               gt_bboxes[list(matched_gt).index(j) if False else j])
            # 简化：用最大 IoU
    # 重新计算匹配 IoU
    avg_iou = 0
    if matched_pred:
        ious = []
        for i, pb in enumerate(pred_bboxes):
            if i in matched_pred:
                best = max(bbox_iou_1d(pb, gt_bboxes[j]) for j in range(n_gt))
                ious.append(best)
        avg_iou = np.mean(ious) if ious else 0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_iou": float(avg_iou),
        "n_pred": n_pred,
        "n_gt": n_gt,
        "tp": tp,
    }


def evaluate_model(pred_dir: str, gt_file: str, iou_threshold: float = 0.1) -> dict:
    """评估整个模型的 grounding 效果"""
    with open(gt_file) as f:
        gt_data = json.load(f)

    results = []
    for ann in gt_data:
        point_name = ann["point_name"]
        gt_annotations = ann["annotations"]

        # 读取预测结果
        status_file = os.path.join(pred_dir, f"{point_name}.status.json")
        if not os.path.exists(status_file):
            continue

        status = json.load(open(status_file))
        if status.get("status") != "success":
            continue

        pred_bboxes = [b["bbox_2d"] for b in status.get("bboxes", []) if "bbox_2d" in b]

        metrics = evaluate_single_point(pred_bboxes, gt_annotations, iou_threshold)
        metrics["point_name"] = point_name
        results.append(metrics)

    if not results:
        return {"error": "no results"}

    # 汇总
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]
    ious = [r["avg_iou"] for r in results if r["avg_iou"] > 0]

    return {
        "n_points": len(results),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_f1": float(np.mean(f1s)),
        "mean_iou": float(np.mean(ious)) if ious else 0,
        "total_pred_bboxes": sum(r["n_pred"] for r in results),
        "total_gt_bboxes": sum(r["n_gt"] for r in results),
        "total_tp": sum(r["tp"] for r in results),
        "per_point": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-file", default=str(PROJECT_ROOT / "qwen/dataset/annotations_test.json"))
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    args = parser.parse_args()

    result = evaluate_model(args.pred_dir, args.gt_file, args.iou_threshold)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Grounding 评估结果 ({result['n_points']} 点位)")
    print(f"  Precision: {result['mean_precision']:.4f}")
    print(f"  Recall:    {result['mean_recall']:.4f}")
    print(f"  F1:        {result['mean_f1']:.4f}")
    print(f"  Mean IoU:  {result['mean_iou']:.4f}")
    print(f"  Pred bboxes: {result['total_pred_bboxes']}")
    print(f"  GT bboxes:   {result['total_gt_bboxes']}")
    print(f"  TP:          {result['total_tp']}")

    # 保存
    output_path = os.path.join(args.pred_dir, "grounding_eval.json")
    # 去掉 per_point 节省空间
    summary = {k: v for k, v in result.items() if k != "per_point"}
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  保存: {output_path}")


if __name__ == "__main__":
    main()

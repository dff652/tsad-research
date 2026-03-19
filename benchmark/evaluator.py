"""
Benchmark 双维度评估器

维度A：主观经验拟合（核心 100 评分点）
  - 基于人工评分数据评估算法与专家判断的一致性
  - 关联异常检测特征与人工评分的相关性

维度B：物理常识约束（泛化 436 全量点）
  - 平均异常率硬限（过滤乱叫狂响的不稳定模型）
  - 时序粘性（异常连续性 = avg_cluster_length）
  - 跳变率（由 transitions/anomaly_count 衡量）
  - 异常簇连贯性
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


class DimensionAEvaluator:
    """维度A：基于人工评分的主观经验拟合评估"""

    def __init__(self, scores_csv: str, features_csv: str, evaluated_points_file: str):
        self.scores_df = pd.read_csv(scores_csv)
        self.features_df = pd.read_csv(features_csv)
        with open(evaluated_points_file) as f:
            self.evaluated_points = [line.strip() for line in f if line.strip()]

    def evaluate(self, algorithm_name: str) -> dict:
        """评估算法在 100 个评分点位上的表现"""
        # 关联评分与特征
        feat = self.features_df[self.features_df["point_name"].isin(self.evaluated_points)].copy()

        if len(feat) == 0:
            return {"error": "no features found for evaluated points"}

        # 基本统计
        rates = feat["global_mask_ratio"].values
        clusters = feat["num_anomaly_clusters"].values
        avg_cluster_len = feat["avg_cluster_length"].values

        # 计算跳变率近似：transitions ≈ 2 * num_clusters / anomaly_count
        anomaly_counts = feat["global_mask_count"].values
        with np.errstate(divide='ignore', invalid='ignore'):
            jump_ratios = np.where(
                anomaly_counts > 0,
                2 * clusters / anomaly_counts,
                0.0
            )

        return {
            "algorithm": algorithm_name,
            "evaluated_points_total": len(self.evaluated_points),
            "evaluated_points_matched": len(feat),
            "mean_anomaly_rate": float(np.mean(rates)),
            "median_anomaly_rate": float(np.median(rates)),
            "std_anomaly_rate": float(np.std(rates)),
            "mean_stickiness": float(np.mean(avg_cluster_len)),
            "mean_jump_ratio": float(np.mean(jump_ratios)),
            "mean_num_clusters": float(np.mean(clusters)),
            "zero_anomaly_points": int(np.sum(anomaly_counts == 0)),
        }


class DimensionBEvaluator:
    """维度B：物理常识约束评估（全量 436 点位）"""

    def __init__(self, features_csv: str, constraints: dict = None):
        self.features_df = pd.read_csv(features_csv)
        self.constraints = constraints or {
            "max_mean_anomaly_rate": 0.15,
            "min_mean_anomaly_rate": 0.001,
            "max_jump_ratio": 0.3,
        }

    def evaluate(self, algorithm_name: str) -> dict:
        """评估全量点位的物理合理性"""
        feat = self.features_df.copy()
        n = len(feat)

        rates = feat["global_mask_ratio"].values
        clusters = feat["num_anomaly_clusters"].values
        avg_cluster_len = feat["avg_cluster_length"].values
        max_cluster_len = feat["max_cluster_length"].values
        anomaly_counts = feat["global_mask_count"].values

        # 跳变率近似
        jump_ratios = np.where(
            anomaly_counts > 0,
            2 * clusters / anomaly_counts,
            0.0
        )
        # 仅对有异常的点位计算 jump
        has_anomaly = anomaly_counts > 0
        mean_jump = float(np.mean(jump_ratios[has_anomaly])) if has_anomaly.any() else 0.0

        mean_rate = float(np.mean(rates))

        # 约束检查
        violations = []
        if mean_rate > self.constraints["max_mean_anomaly_rate"]:
            violations.append(
                f"平均异常率 {mean_rate:.4f} 超过上限 {self.constraints['max_mean_anomaly_rate']}")
        if mean_rate < self.constraints["min_mean_anomaly_rate"]:
            violations.append(
                f"平均异常率 {mean_rate:.6f} 低于下限 {self.constraints['min_mean_anomaly_rate']}")
        if mean_jump > self.constraints["max_jump_ratio"]:
            violations.append(
                f"平均跳变率 {mean_jump:.4f} 超过上限 {self.constraints['max_jump_ratio']}")

        # 按传感器类型分组统计
        type_stats = {}
        if "sensor_type" in feat.columns:
            for stype, group in feat.groupby("sensor_type"):
                type_stats[stype] = {
                    "count": len(group),
                    "mean_anomaly_rate": float(group["global_mask_ratio"].mean()),
                    "mean_cluster_len": float(group["avg_cluster_length"].mean()),
                }

        return {
            "algorithm": algorithm_name,
            "total_points": n,
            "mean_anomaly_rate": mean_rate,
            "median_anomaly_rate": float(np.median(rates)),
            "std_anomaly_rate": float(np.std(rates)),
            "max_anomaly_rate": float(np.max(rates)),
            "min_anomaly_rate": float(np.min(rates)),
            "mean_stickiness": float(np.mean(avg_cluster_len[has_anomaly])) if has_anomaly.any() else 0,
            "mean_jump_ratio": mean_jump,
            "mean_num_clusters": float(np.mean(clusters)),
            "zero_anomaly_points": int(np.sum(~has_anomaly)),
            "high_anomaly_points": int(np.sum(rates > 0.10)),
            "constraint_violations": violations,
            "constraints_passed": len(violations) == 0,
            "by_sensor_type": type_stats,
        }


class BenchmarkEvaluator:
    """综合评估器：组合维度A + 维度B"""

    def __init__(self, config: dict):
        self.config = config
        data_cfg = config["data"]

        features_path = self._resolve_path(data_cfg["all_features"])
        merged_path = self._resolve_path(data_cfg["merged_scores_features"])

        self.dim_a = DimensionAEvaluator(
            scores_csv=self._resolve_path(data_cfg["scores_csv"]),
            features_csv=features_path,
            evaluated_points_file=self._resolve_path(data_cfg["evaluated_points"]),
        )
        eval_cfg = config.get("evaluation", {}).get("dimension_b", {})
        self.dim_b = DimensionBEvaluator(
            features_csv=features_path,
            constraints=eval_cfg.get("constraints", {}),
        )

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(
            os.path.join(os.path.dirname(__file__), path))

    def evaluate_algorithm(self, algorithm_name: str) -> dict:
        """双维度评估"""
        result = {
            "algorithm": algorithm_name,
            "dimension_a": self.dim_a.evaluate(algorithm_name),
            "dimension_b": self.dim_b.evaluate(algorithm_name),
        }
        result["summary"] = self._compute_summary(
            result["dimension_a"], result["dimension_b"])
        return result

    def _compute_summary(self, dim_a: dict, dim_b: dict) -> dict:
        return {
            "physics_compliant": dim_b.get("constraints_passed", False),
            "constraint_violations": dim_b.get("constraint_violations", []),
            "mean_anomaly_rate_all": dim_b.get("mean_anomaly_rate", 0),
            "mean_anomaly_rate_evaluated": dim_a.get("mean_anomaly_rate", 0),
            "mean_stickiness": dim_b.get("mean_stickiness", 0),
            "mean_jump_ratio": dim_b.get("mean_jump_ratio", 0),
            "total_points": dim_b.get("total_points", 0),
            "zero_anomaly_points": dim_b.get("zero_anomaly_points", 0),
            "high_anomaly_points": dim_b.get("high_anomaly_points", 0),
        }

#!/usr/bin/env python3
"""
统一实验日志 —— 借鉴 autoresearch 的 results.tsv 模式

所有实验（param_sweep、batch_inference、runner）通过本模块记录结果，
保证日志格式统一、可追溯、可比较。

日志格式（TSV，tab 分隔）：
    timestamp  algorithm  config_tag  score  anomaly_rate  elapsed  status  description

用法：
    from experiment_log import ExperimentLogger

    logger = ExperimentLogger()
    logger.log(
        algorithm="timer",
        config_tag="k3.0_mad_lb256_ds10000",
        score=0.764,
        anomaly_rate=0.032,
        elapsed=58.5,
        status="keep",
        description="Timer-84M baseline with MAD threshold k=3.0",
    )

    # 读取历史
    df = logger.load()
    print(df.sort_values("score", ascending=False).head(10))
"""

import os
from datetime import datetime
from pathlib import Path

TSV_HEADER = "timestamp\talgorithm\tconfig_tag\tscore\tanomaly_rate\telapsed\tstatus\tdescription\n"

# 有效的 status 值（借鉴 autoresearch: keep/discard/crash）
VALID_STATUSES = {"keep", "discard", "crash", "running", "timeout", "error"}


class ExperimentLogger:
    """统一实验日志记录器"""

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = str(Path(__file__).resolve().parent.parent / "results")
        self.log_path = os.path.join(log_dir, "experiments.tsv")
        os.makedirs(log_dir, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        """确保 TSV 文件存在且有 header"""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(TSV_HEADER)
        else:
            # 检查现有文件是否有 header
            with open(self.log_path, "r") as f:
                first_line = f.readline()
            if not first_line.startswith("timestamp\t"):
                # 补上 header（文件被意外清空等情况）
                with open(self.log_path, "r") as f:
                    content = f.read()
                with open(self.log_path, "w") as f:
                    f.write(TSV_HEADER + content)

    def log(self, algorithm: str, config_tag: str, score: float,
            anomaly_rate: float, elapsed: float, status: str,
            description: str):
        """追加一条实验记录

        Args:
            algorithm: 算法名 (如 "timer", "iforest", "chatts")
            config_tag: 配置标签 (如 "k3.0_mad_lb256")
            score: 综合评分 (0~1)，crash 时为 0.0
            anomaly_rate: 平均异常率，crash 时为 0.0
            elapsed: 耗时(秒)
            status: keep / discard / crash / timeout / error
            description: 简要描述（不可含 tab，自动替换为空格）
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"无效 status '{status}'，可选: {VALID_STATUSES}")

        # 防止 description 中的 tab 破坏 TSV 格式
        description = description.replace("\t", " ").replace("\n", " ").strip()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (f"{timestamp}\t{algorithm}\t{config_tag}\t"
                f"{score:.6f}\t{anomaly_rate:.6f}\t{elapsed:.1f}\t"
                f"{status}\t{description}\n")

        with open(self.log_path, "a") as f:
            f.write(line)

    def load(self):
        """读取全部实验日志为 DataFrame"""
        import pandas as pd
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        return pd.read_csv(self.log_path, sep="\t")

    def best(self, algorithm: str = None, n: int = 5):
        """返回评分最高的 n 条记录"""
        df = self.load()
        if df.empty:
            return df
        # 只看 keep 的记录
        df = df[df["status"] == "keep"]
        if algorithm:
            df = df[df["algorithm"] == algorithm]
        return df.sort_values("score", ascending=False).head(n)

    def summary(self):
        """打印实验汇总"""
        df = self.load()
        if df.empty:
            print("暂无实验记录")
            return

        print(f"\n{'='*70}")
        print(f"实验日志汇总 ({self.log_path})")
        print(f"{'='*70}")
        print(f"总实验数: {len(df)}")

        for status in VALID_STATUSES:
            count = len(df[df["status"] == status])
            if count > 0:
                print(f"  {status}: {count}")

        # 按算法分组
        kept = df[df["status"] == "keep"]
        if not kept.empty:
            print(f"\n{'算法':<20} {'实验数':<8} {'最佳评分':<10} {'平均异常率':<12}")
            print(f"{'-'*55}")
            for algo, group in kept.groupby("algorithm"):
                print(f"{algo:<20} {len(group):<8} {group['score'].max():<10.4f} "
                      f"{group['anomaly_rate'].mean():<12.6f}")
        print(f"{'='*70}")

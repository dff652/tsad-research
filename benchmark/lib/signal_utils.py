"""
信号处理工具库 —— 从 ts-iteration-loop 抽取的核心函数

抽取原则：仅复制本项目实际用到的函数，解除对项目 2 的路径依赖。
原始来源：/home/douff/ts/ts-iteration-loop/services/inference/signal_processing.py
"""

import numpy as np
import pandas as pd


def ts_downsample(data, downsampler='m4', n_out=100000):
    """
    Downsample time series data

    Args:
        data: pd.Series - 输入时间序列数据
        downsampler: str - 降采样方法 ('m4' 或 'minmax')
        n_out: int - 输出数据点数

    Returns:
        tuple of (downsampled_data, downsampled_time, position_index)
    """
    from tsdownsample import MinMaxLTTBDownsampler, M4Downsampler

    if downsampler == 'm4':
        s_ds = M4Downsampler().downsample(data.values, n_out=n_out)
    elif downsampler == 'minmax':
        s_ds = MinMaxLTTBDownsampler().downsample(data.values, n_out=n_out)
    else:
        raise ValueError(f"Unsupported downsampler: {downsampler}")

    downsampled_data = data.iloc[s_ds]
    downsampled_time = data.index[s_ds]
    position_index = np.asarray(s_ds, dtype=np.int64)

    return downsampled_data, downsampled_time, position_index


def ts_downsample_numpy(values: np.ndarray, n_out: int = 10000) -> tuple:
    """
    纯 numpy M4 降采样（不依赖 tsdownsample 库）

    每段保留 min, max, first, last 四个点。

    Args:
        values: 1D numpy array
        n_out: 目标点数

    Returns:
        (downsampled_values, position_indices)
    """
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
            idx_base,                          # first
            idx_base + int(np.argmin(chunk)),   # min
            idx_base + int(np.argmax(chunk)),   # max
            idx_base + len(chunk) - 1,          # last
        ])
    indices = sorted(set(indices))
    return values[indices], np.array(indices)

"""
@File    :   timer_detect.py
@Time    :   2024/12/09
@Author  :   DouFengfeng
@Desc    :   基于 Timer/SunDial 大模型的时序异常检测

Timer 是清华大学开源的时序预测模型，本模块将其封装为可在 run.py 中调用的异常检测方法。
通过滚动预测计算残差，基于 MAD/Sigma 方法检测异常区间。
注意：该方法需要 GPU 资源。
"""

import time
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import pandas as pd

# GPU 相关依赖（懒加载）
_torch = None


def _lazy_import_torch():
    """懒加载 torch，避免无 GPU 环境下报错"""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


TensorLike = Union["torch.Tensor", "np.ndarray"]
DetectorFunc = None  # Callable[[torch.Tensor], List[dict]]


class TimerAnomalyPipeline:
    """
    TIMER/SunDial 推理 + 残差异常检测一体化管线。

    设计目标：
    1. 一次加载模型，在多次预测/检测中复用；
    2. 支持 GPU/CPU 自由切换；
    3. 提供端到端接口：输入历史序列 → 返回预测、残差、异常区间。
    """

    def __init__(
        self,
        model_path: str,
        device: Union[str, dict] = "cpu",
        trust_remote_code: bool = True,
    ) -> None:
        """
        Args:
            model_path: 预训练权重路径或 HF Repo id。
            device: 
                - 单卡/CPU: "cpu" 或 "cuda:0" 等字符串
                - 多卡: "auto" 让 Hugging Face 自动分配，或传入 device_map 字典手动指定
            trust_remote_code: 是否加载自定义模型代码。
        """
        torch = _lazy_import_torch()
        from transformers import AutoModelForCausalLM
        
        self.device_str = device
        self.is_multi_gpu = device == "auto" or (isinstance(device, dict))
        
        if self.is_multi_gpu:
            # 多卡模式：使用 device_map
            device_map = device if isinstance(device, dict) else "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
            )
            # 多卡时，输入数据通常放在第一张 GPU（cuda:0）
            self.input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # 单卡/CPU 模式：使用 .to(device)
            self.device = torch.device(device)
            self.input_device = self.device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            ).to(self.device)
        
        self.model.eval()

    def rolling_forecast_residuals(
        self,
        data: TensorLike,
        *,
        lookback_length: int,
        streaming: bool = True,
        forecast_horizon: int = 1,
        reset_interval: int = 256,
        num_samples: Optional[int] = None,
    ) -> "torch.Tensor":
        """
        对一维序列做逐点前视预测，返回真实值与预测值的残差序列。

        Args:
            data: 1D 数据（torch.Tensor 或 numpy.ndarray）。
            lookback_length: 滚动窗口长度，要求 < len(data)。
            streaming: True 表示启用"累加 + 周期重置"的流式模式；
                False 表示每次直接切片最近窗口（适合离线批处理）。
            forecast_horizon: 预测步长（未来生成的步数），需 ≥1。
            reset_interval: 每跑多少步重置一次上下文（使用真实窗口），避免误差积累。
            num_samples: 若模型支持多样本，则对样本取均值作为预测。

        Returns:
            torch.Tensor: 残差矩阵，形状 (len(data) - lookback_length - forecast_horizon + 1, forecast_horizon)。
        """
        torch = _lazy_import_torch()

        if isinstance(data, np.ndarray):
            series = torch.as_tensor(data, dtype=torch.float32, device=self.input_device)
        else:
            series = data.to(self.input_device).float()

        assert series.dim() == 1, "data 必须是一维序列"
        T = int(series.shape[0])
        assert T > lookback_length, "data 长度需大于 lookback_length"
        assert forecast_horizon >= 1, "forecast_horizon 需 ≥ 1"
        assert lookback_length + forecast_horizon <= T, "窗口+步长不能超过序列长度"

        residuals: List[torch.Tensor] = []
        if streaming:
            context = series[:lookback_length].clone()
            steps_since_reset = 0
        end_t = T - forecast_horizon

        with torch.inference_mode():
            for t in range(lookback_length, end_t + 1):
                if streaming:
                    if steps_since_reset >= reset_interval:
                        context = series[t - lookback_length:t].clone()
                        steps_since_reset = 0
                    seq = context[-lookback_length:].unsqueeze(0)
                else:
                    window = series[t - lookback_length:t]
                    seq = window.unsqueeze(0)
                
                # 检查输入序列是否包含异常值或全零
                seq_std = seq.std().item()
                seq_mean = seq.mean().item()
                
                if torch.any(~torch.isfinite(seq)):
                    y_hat_block = torch.full((forecast_horizon,), float('nan'), 
                                            device=seq.device, dtype=seq.dtype)
                elif seq_std < 1e-8:
                    last_value = seq[0, -1].item() if seq.shape[1] > 0 else seq_mean
                    y_hat_block = torch.full((forecast_horizon,), last_value,
                                            device=seq.device, dtype=seq.dtype)
                else:
                    gen_kwargs = dict(max_new_tokens=forecast_horizon, use_cache=False)
                    if num_samples is not None:
                        gen_kwargs["num_samples"] = num_samples
                    pred = self.model.generate(seq, **gen_kwargs)
                    
                    if torch.any(~torch.isfinite(pred)):
                        last_value = seq[0, -1].item()
                        y_hat_block = torch.full((forecast_horizon,), last_value,
                                                device=seq.device, dtype=seq.dtype)
                    else:
                        y_hat_block = self._extract_forecast(pred, forecast_horizon)

                y_true_block = series[t : t + forecast_horizon]
                residual_block = (y_true_block - y_hat_block).detach()
                residuals.append(residual_block)

                if streaming:
                    context = torch.cat([context, series[t].view(1)], dim=0)
                    steps_since_reset += 1

        res_tensor = torch.stack(residuals, dim=0)
        if forecast_horizon == 1:
            return res_tensor.squeeze(-1)
        return res_tensor

    def detect_anomalies_from_residuals(
        self,
        residuals: "torch.Tensor",
        *,
        method: str = "mad",
        residual_step: int = 0,
        threshold_k: float = 3.5,
        min_run: int = 1,
    ) -> List[dict]:
        """
        使用指定策略对残差做异常检测，并输出连续区间。

        Args:
            residuals: 1D 残差张量。
            method: "mad"（默认）、"sigma"（3-sigma）。
            residual_step: 若残差为 2D（多步预测），选择第几步的残差参与检测。
            threshold_k: 阈值系数。
            min_run: 至少连续多少个点才算异常区间。

        Returns:
            List[dict]: [{ "range": (start, end), "score": float }, ...]
        """
        torch = _lazy_import_torch()
        
        if residuals.dim() == 2:
            assert 0 <= residual_step < residuals.shape[1], "residual_step 越界"
            r = residuals[:, residual_step].float()
        else:
            r = residuals.float()
        assert r.dim() == 1, "residuals 必须是一维或二维张量"
        method = method.lower()

        if method == "mad":
            center = torch.median(r)
            scale = torch.median(torch.abs(r - center))
            if scale <= 1e-8:
                scale = torch.std(r)
        elif method == "sigma":
            center = torch.mean(r)
            scale = torch.std(r)
        else:
            raise ValueError("method 仅支持 'mad' 或 'sigma'")

        if scale <= 0 or not torch.isfinite(scale):
            scale = torch.tensor(1.0, device=r.device)

        z = torch.abs(r - center) / scale
        mask = (z > threshold_k)

        ranges: List[dict] = []
        start: Optional[int] = None
        for idx in range(len(mask)):
            if mask[idx]:
                if start is None:
                    start = idx
            else:
                if start is not None and idx - start >= min_run:
                    score = float(z[start:idx].max().item())
                    ranges.append({"range": (start, idx - 1), "score": score})
                start = None
        if start is not None and len(mask) - start >= min_run:
            score = float(z[start:].max().item())
            ranges.append({"range": (start, len(mask) - 1), "score": score})
        return ranges

    def detect_series(
        self,
        data: TensorLike,
        *,
        lookback_length: int,
        streaming: bool = True,
        reset_interval: int = 256,
        num_samples: Optional[int] = None,
        forecast_horizon: int = 1,
        residual_step: int = 0,
        method: str = "mad",
        threshold_k: float = 3.5,
        min_run: int = 1,
    ) -> Tuple["torch.Tensor", List[dict]]:
        """
        端到端异常检测：输入历史序列 → 输出残差和异常区间。

        Returns:
            (residuals, intervals)
        """
        residuals = self.rolling_forecast_residuals(
            data,
            lookback_length=lookback_length,
            streaming=streaming,
            reset_interval=reset_interval,
            num_samples=num_samples,
            forecast_horizon=forecast_horizon,
        )

        intervals = self.detect_anomalies_from_residuals(
            residuals,
            method=method,
            residual_step=residual_step,
            threshold_k=threshold_k,
            min_run=min_run,
        )
        return residuals, intervals

    def _extract_forecast(self, generated: "torch.Tensor", horizon: int) -> "torch.Tensor":
        """从 generate 输出中抽取最近 horizon 步的预测值。"""
        torch = _lazy_import_torch()
        out = generated
        
        if out.dim() == 3:
            if out.shape[1] > 1:
                out = out.mean(dim=1)
            else:
                out = out.mean(dim=0)
        
        if out.dim() == 2:
            if out.shape[0] == 0:
                return torch.full((horizon,), float('nan'), device=out.device, dtype=out.dtype)
            if out.shape[1] < horizon:
                result = out[0, :].clone()
                padding = torch.full((horizon - out.shape[1],), float('nan'), 
                                    device=out.device, dtype=out.dtype)
                return torch.cat([result, padding])
            return out[0, -horizon:]
        
        flat = out.flatten()
        if len(flat) < horizon:
            padding = torch.full((horizon - len(flat),), float('nan'), 
                               device=flat.device, dtype=flat.dtype)
            return torch.cat([flat, padding])
        
        return flat[-horizon:]


# ============================================================================
# 辅助函数
# ============================================================================

def map_anomalies_to_original(
    anomalies: List[Dict], 
    position_index: np.ndarray,
    lookback_length: int = 0,
) -> List[Dict]:
    """
    将降采样序列上的异常索引映射到原始 DataFrame 索引。
    
    Args:
        anomalies: 检测返回的异常列表，每个元素包含 "range": (start, end)
        position_index: ts_downsample 返回的 position_index（整数位置索引数组）
        lookback_length: 残差索引偏移量（残差从 lookback_length 位置开始）
    
    Returns:
        映射后的异常列表，range 变为原始数据的位置索引
    """
    idx_array = np.asarray(position_index)
    
    mapped = []
    for a in anomalies:
        ds_start, ds_end = a["range"]
        
        # 残差索引 + lookback_length = 降采样序列索引
        ds_start_adj = ds_start + lookback_length
        ds_end_adj = ds_end + lookback_length
        
        # 边界检查
        ds_start_adj = max(0, min(ds_start_adj, len(idx_array) - 1))
        ds_end_adj = max(0, min(ds_end_adj, len(idx_array) - 1))
        
        # 通过 position_index 映射到原始位置索引
        orig_start = int(idx_array[ds_start_adj])
        orig_end = int(idx_array[ds_end_adj])
        
        mapped_anomaly = a.copy()
        mapped_anomaly["range"] = [orig_start, orig_end]
        mapped_anomaly["downsampled_range"] = [a["range"][0], a["range"][1]]
        mapped.append(mapped_anomaly)
    
    return mapped


def create_mask_from_anomalies(
    data_length: int, 
    anomalies: List[Dict]
) -> np.ndarray:
    """
    根据异常区间列表创建布尔掩码
    
    Args:
        data_length: 原始数据长度
        anomalies: 异常列表，每个元素包含 "range": [start, end]
        
    Returns:
        与原始数据等长的整数掩码（0=正常，1=异常）
    """
    mask = np.zeros(data_length, dtype=int)
    for a in anomalies:
        rng = a["range"]
        # 支持 tuple 和 list 两种格式
        start = rng[0] if isinstance(rng, (list, tuple)) else rng["start"]
        end = rng[1] if isinstance(rng, (list, tuple)) else rng["end"]
        start = max(0, min(start, data_length - 1))
        end = max(0, min(end, data_length - 1))
        mask[start:end+1] = 1
    return mask


# ============================================================================
# 单例模式管理模型实例
# ============================================================================

_pipeline_instance: Optional[TimerAnomalyPipeline] = None
_pipeline_config: Optional[Dict] = None


def get_timer_pipeline(
    model_path: str,
    device: str = "cuda:0",
) -> TimerAnomalyPipeline:
    """
    获取 TimerAnomalyPipeline 实例（单例模式）
    
    如果配置相同则复用现有实例，否则重新创建
    """
    global _pipeline_instance, _pipeline_config
    
    new_config = {"model_path": model_path, "device": device}
    
    if _pipeline_instance is None or _pipeline_config != new_config:
        print(f"[Timer] 正在加载模型: {model_path} 到 {device}...")
        _pipeline_instance = TimerAnomalyPipeline(
            model_path=model_path,
            device=device,
        )
        _pipeline_config = new_config
        print("[Timer] 模型加载完成")
    
    return _pipeline_instance


# ============================================================================
# 主检测函数
# ============================================================================

def timer_detect(
    data: pd.DataFrame,
    model_path: str,
    device: str = "cuda:0",
    n_downsample: int = 10000,
    downsampler: str = "m4",
    lookback_length: int = 256,
    threshold_k: float = 3.5,
    method: str = "mad",
    min_run: int = 1,
    streaming: bool = False,
    reset_interval: int = 256,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, List[Dict], Optional[np.ndarray]]:
    """
    使用 Timer 进行异常检测
    
    Args:
        data: 输入数据，DataFrame 格式，第一列为时序值
        model_path: Timer 模型路径
        device: GPU 设备（如 "cuda:0"）
        n_downsample: 降采样点数
        downsampler: 降采样方法（'m4' 或 'minmax'）
        lookback_length: 滚动预测窗口长度
        threshold_k: 异常检测阈值系数
        method: 残差检测方法（'mad' 或 'sigma'）
        min_run: 最小连续异常点数
        streaming: 是否使用流式模式
        reset_interval: 流式模式下的上下文重置周期
        forecast_horizon: 预测步长
    
    Returns:
        global_mask: 与原始数据对齐的异常掩码（0=正常，1=异常）
        anomalies: 异常区间详细信息列表（已映射到原始索引）
        position_index: 本次检测所用的降采样位置索引
    """
    from signal_utils import ts_downsample
    
    # 获取列名和数据
    column = data.columns[0]
    series = data[column]
    data_length = len(series)
    
    # 降采样
    if downsampler is None or str(downsampler).lower() == "none":
        ts_values = series.values.astype(np.float32)
        position_index = np.arange(data_length)
        print(f"[Timer] downsampler=none，跳过降采样")
    elif data_length > n_downsample:
        downsampled_data, _, position_index = ts_downsample(
            series, downsampler=downsampler, n_out=n_downsample
        )
        ts_values = downsampled_data.values.astype(np.float32)
        print(f"[Timer] 降采样: {data_length} -> {len(ts_values)} 点")
    else:
        ts_values = series.values.astype(np.float32)
        position_index = np.arange(data_length)
        print(f"[Timer] 数据长度 {data_length} <= {n_downsample}，不进行降采样")
    
    # 检查数据长度是否满足 lookback_length 要求
    if len(ts_values) <= lookback_length:
        print(f"[Timer] 警告: 数据长度 {len(ts_values)} <= lookback_length {lookback_length}，无法检测")
        return np.zeros(data_length, dtype=int), [], position_index
    
    # 获取 pipeline 实例
    pipeline = get_timer_pipeline(model_path, device)
    
    # 执行推理
    st = time.time()
    residuals, intervals = pipeline.detect_series(
        ts_values,
        lookback_length=lookback_length,
        streaming=streaming,
        reset_interval=reset_interval,
        forecast_horizon=forecast_horizon,
        method=method,
        threshold_k=threshold_k,
        min_run=min_run,
    )
    et = time.time()
    print(f"[Timer] 推理耗时: {et - st:.2f}s, 检测到 {len(intervals)} 个异常区间")
    
    # 映射到原始索引
    if intervals:
        mapped_anomalies = map_anomalies_to_original(
            intervals, position_index, lookback_length=lookback_length
        )
    else:
        mapped_anomalies = []
    
    # 创建掩码
    global_mask = create_mask_from_anomalies(data_length, mapped_anomalies)
    
    return global_mask, mapped_anomalies, position_index

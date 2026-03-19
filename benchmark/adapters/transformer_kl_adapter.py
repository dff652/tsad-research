#!/usr/bin/env python3
"""
Transformer+KL 自监督异常检测适配器

原理（来自 ilabel/cv_detect/anml_trsf.py）：
1. 用噪声先验（粉红噪声）训练 Transformer 重建模型
2. 引入注意力 KL 散度作为正则化（期望注意力符合局部性先验）
3. 在真实数据上评估：注意力偏离先验的程度 = 异常分数

这是表示学习+自监督范式，不依赖预测残差。
训练集：337 个非评分点位的数据（可选）
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ===== 模型定义（来自 anml_trsf.py）=====

class WindowDataset(Dataset):
    def __init__(self, series, window):
        self.series = series.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.series) - self.window

    def __getitem__(self, idx):
        seq = self.series[idx: idx + self.window]
        return torch.tensor(seq).unsqueeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_lin(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_lin(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_lin(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_lin(out), attn


class EncoderBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights


class TransfKL(nn.Module):
    def __init__(self, seq_len=48, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, 1)
        self.register_buffer("prior", self._make_prior(seq_len))

    def _make_prior(self, L, lam=3.0):
        pos = torch.arange(L).unsqueeze(1)
        dist = (pos - pos.T).abs().float()
        prior = torch.exp(-dist / lam)
        return prior / prior.sum(dim=-1, keepdim=True)

    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        attentions = []
        for layer in self.layers:
            h, attn = layer(h)
            attentions.append(attn)
        return self.output_proj(h), attentions


def kl_series_prior(p, q, eps=1e-9):
    p = torch.clamp(p, eps, 1)
    q = torch.clamp(q, eps, 1)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


# ===== 训练和推理 =====

def generate_pink_noise(length: int) -> np.ndarray:
    """生成粉红噪声作为训练先验"""
    t = np.linspace(0, 1, length)
    f = np.fft.fftfreq(length, t[1] - t[0])
    f[0] = 1
    pink_spectrum = 1 / np.sqrt(np.abs(f))
    pink_noise = np.real(np.fft.ifft(pink_spectrum * np.exp(1j * np.random.uniform(0, 2 * np.pi, length))))
    pink_noise = (pink_noise - pink_noise.min()) / (pink_noise.max() - pink_noise.min() + 1e-8)
    return pink_noise


def train_model(window: int = 48, d_model: int = 64, n_heads: int = 4,
                n_layers: int = 2, epochs: int = 32, lr: float = 2e-4,
                lambda_kl: float = 0.1) -> TransfKL:
    """用噪声先验训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = generate_pink_noise(65536)
    dataset = WindowDataset(noise, window)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    model = TransfKL(seq_len=window, d_model=d_model, n_heads=n_heads,
                     n_layers=n_layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optim.zero_grad()
            recon, atts = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            kl_loss = kl_series_prior(atts[-1], model.prior.to(device)).mean()
            loss = recon_loss + lambda_kl * kl_loss
            loss.backward()
            optim.step()
            losses.append(loss.item())

    return model


def detect_anomalies(model, values: np.ndarray, window: int = 48,
                      threshold_percentile: float = 95) -> tuple:
    """用训练好的模型检测异常"""
    device = next(model.parameters()).device
    model.eval()

    # 标准化
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-8:
        return np.zeros(len(values), dtype=int), np.zeros(len(values))

    normalized = (values - v_min) / (v_max - v_min)

    scores = np.zeros(len(values))
    count = np.zeros(len(values))

    with torch.no_grad():
        for i in range(len(normalized) - window):
            seq = torch.tensor(normalized[i:i + window]).unsqueeze(0).unsqueeze(-1).float().to(device)
            recon, atts = model(seq)

            # KL 异常分数：注意力偏离先验的程度
            kl = kl_series_prior(atts[-1], model.prior.to(device)).cpu().numpy()[0]
            kl_score = np.abs(kl).max(axis=(0, 1))  # 取最大偏离

            # 重建误差
            recon_err = abs(normalized[i + window - 1] - recon[0, -1, 0].item())

            score = recon_err + 0.1 * kl_score
            scores[i + window - 1] += score
            count[i + window - 1] += 1

    count[count == 0] = 1
    scores /= count

    threshold = np.percentile(scores[scores > 0], threshold_percentile) if (scores > 0).any() else 0
    mask = (scores > threshold).astype(int)

    return mask, scores


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--n-downsample", type=int, default=10000)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=32)
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

        print(f"[TransfKL] 点位: {args.point_name}, 数据量: {len(values)}")

        if len(values) > args.n_downsample:
            ds_values, ds_indices = downsample_m4(values, args.n_downsample)
        else:
            ds_values, ds_indices = values, np.arange(len(values))

        # 训练（每个点位独立训练，用噪声先验）
        st = time.time()
        model = train_model(window=args.window, epochs=args.epochs)
        train_time = time.time() - st

        # 检测
        st2 = time.time()
        mask_ds, scores_ds = detect_anomalies(model, ds_values, window=args.window)
        detect_time = time.time() - st2

        # 映射回原始长度
        if len(ds_values) < len(values):
            mask = np.zeros(len(values), dtype=int)
            for i, idx in enumerate(ds_indices):
                if mask_ds[i] == 1:
                    mask[idx] = 1
        else:
            mask = mask_ds

        rate = float(mask.mean())
        print(f"[TransfKL] 完成: rate={rate:.4f}, train={train_time:.1f}s, detect={detect_time:.1f}s")

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.compact:
            pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(args.output, index=False)
        else:
            pd.DataFrame({"Time": df["Time"], "value": df[value_col],
                          "global_mask": mask.astype(int)}).to_csv(args.output, index=False)

        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({"status": "success", "anomaly_rate": rate,
                       "anomaly_count": int(mask.sum()), "total_rows": len(values),
                       "train_time": train_time, "detect_time": detect_time}, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output.replace(".csv", ".status.json"), "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

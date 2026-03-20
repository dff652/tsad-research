#!/usr/bin/env python3
"""
GAF + ConvAutoencoder 异常检测

流程：
1. 时序 → 滑动窗口切片
2. 每个窗口 → GAF 图像（Gramian Angular Field，保留数值关系）
3. ConvAutoencoder 重建 GAF 图像
4. 重建误差 > 阈值 = 异常窗口
5. 异常窗口映射回原始索引

训练策略：
- 用训练集 337 点位的正常数据训练通用 ConvAE
- 或用粉红噪声/合成数据预训练
- 测试集 99 点位仅推理评估
"""

import os
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ===== GAF 转换 =====

def ts_to_gaf(series: np.ndarray, image_size: int = 64) -> np.ndarray:
    """将一维时序转为 GAF 图像（不依赖 pyts，纯 numpy 实现）"""
    # 1. 缩放到 [-1, 1]
    s_min, s_max = series.min(), series.max()
    if s_max - s_min < 1e-10:
        scaled = np.zeros_like(series)
    else:
        scaled = 2 * (series - s_min) / (s_max - s_min) - 1
    scaled = np.clip(scaled, -1, 1)

    # 2. 如果长度不等于 image_size，用分段聚合缩放
    if len(scaled) != image_size:
        indices = np.linspace(0, len(scaled) - 1, image_size).astype(int)
        scaled = scaled[indices]

    # 3. 转为角度
    phi = np.arccos(scaled)

    # 4. GASF (Gramian Angular Summation Field)
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf.astype(np.float32)


# ===== ConvAutoencoder =====

class ConvAutoencoder(nn.Module):
    """轻量级卷积自编码器，用于 GAF 图像重建"""

    def __init__(self, image_size: int = 64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 64->32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8->4
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # 32->64
            nn.Tanh(),  # GAF 值域 [-1, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class GAFDataset(Dataset):
    """GAF 窗口数据集"""

    def __init__(self, gaf_images: list):
        self.images = gaf_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]).unsqueeze(0)  # (1, H, W)


# ===== 训练 =====

def train_convae(train_gafs: list, image_size: int = 64,
                  epochs: int = 30, lr: float = 1e-3, batch_size: int = 32,
                  device: str = "cuda:1") -> ConvAutoencoder:
    """训练 ConvAutoencoder"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(image_size).to(device)
    dataset = GAFDataset(train_gafs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(epochs):
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1}/{epochs}, loss={np.mean(losses):.6f}")

    return model


def compute_anomaly_scores(model, gafs: list, device: str = "cuda:1") -> np.ndarray:
    """计算每个 GAF 窗口的重建误差"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    scores = []

    with torch.no_grad():
        for gaf in gafs:
            x = torch.tensor(gaf).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
            recon = model(x)
            error = F.mse_loss(recon, x).item()
            scores.append(error)

    return np.array(scores)


# ===== 主流程 =====

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


def extract_windows(values: np.ndarray, window_size: int = 64,
                     step: int = 32) -> list:
    """滑动窗口切片"""
    windows = []
    starts = []
    for i in range(0, len(values) - window_size + 1, step):
        windows.append(values[i:i + window_size])
        starts.append(i)
    return windows, starts


def generate_train_gafs(data_dir: str, points_file: str,
                         window_size: int = 64, n_downsample: int = 5000,
                         max_windows: int = 50000) -> list:
    """从训练集点位生成 GAF 训练数据"""
    import glob, re

    with open(points_file) as f:
        points = [l.strip() for l in f if l.strip()]

    all_gafs = []
    for pt in points:
        matches = glob.glob(os.path.join(data_dir, f"global_adtk_hbos_*_{pt}_*_trend_seasonal_resid.csv"))
        if not matches:
            continue

        df = pd.read_csv(matches[0])
        skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        vcol = [c for c in df.columns if c not in skip][0]
        values = df[vcol].values.astype(np.float64)
        nan_mask = np.isnan(values)
        if nan_mask.any():
            values[nan_mask] = np.nanmedian(values)

        if len(values) > n_downsample:
            values, _ = downsample_m4(values, n_downsample)

        # 提取正常窗口（global_mask=0 的区域）
        # 简化：使用所有窗口（假设大部分数据正常）
        windows, _ = extract_windows(values, window_size, window_size // 2)
        for w in windows:
            all_gafs.append(ts_to_gaf(w, window_size))

        if len(all_gafs) >= max_windows:
            break

    print(f"生成 {len(all_gafs)} 个训练 GAF 窗口（from {len(points)} 点位）")
    return all_gafs[:max_windows]


def detect_single_point(model, csv_path: str, point_name: str,
                         window_size: int = 64, n_downsample: int = 5000,
                         threshold_percentile: float = 95,
                         device: str = "cuda:1") -> tuple:
    """对单个点位进行 GAF 异常检测"""
    df = pd.read_csv(csv_path)
    skip = {"Time", "global_mask", "outlier_mask", "local_mask",
            "global_mask_cluster", "local_mask_cluster"}
    vcol = [c for c in df.columns if c not in skip][0]
    values = df[vcol].values.astype(np.float64)
    nan_mask = np.isnan(values)
    if nan_mask.any():
        values[nan_mask] = np.nanmedian(values)

    original_length = len(values)

    if original_length > n_downsample:
        ds_values, ds_indices = downsample_m4(values, n_downsample)
    else:
        ds_values, ds_indices = values, np.arange(original_length)

    # 提取窗口 + GAF
    step = window_size // 2
    windows, starts = extract_windows(ds_values, window_size, step)
    gafs = [ts_to_gaf(w, window_size) for w in windows]

    # 计算异常分数
    scores = compute_anomaly_scores(model, gafs, device)

    # 阈值
    threshold = np.percentile(scores, threshold_percentile)
    anomaly_windows = scores > threshold

    # 映射回降采样后的 mask
    mask_ds = np.zeros(len(ds_values), dtype=int)
    for i, is_anomaly in enumerate(anomaly_windows):
        if is_anomaly:
            s = starts[i]
            e = min(s + window_size, len(ds_values))
            mask_ds[s:e] = 1

    # 映射回原始长度
    if len(ds_values) < original_length:
        mask = np.zeros(original_length, dtype=int)
        for i, idx in enumerate(ds_indices):
            if mask_ds[i] == 1:
                mask[idx] = 1
    else:
        mask = mask_ds

    return mask, float(mask.mean())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["train", "detect", "full"])
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-downsample", type=int, default=5000)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model-path", default=None, help="已训练模型路径")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = str(PROJECT_ROOT / "data/adtk_hbos_old")
    train_points = str(PROJECT_ROOT / "data/cleaned/train_points.txt")
    test_points_file = str(PROJECT_ROOT / "data/cleaned/evaluated_points.txt")
    output_dir = str(PROJECT_ROOT / "results/predictions/gaf_convae")
    model_save_path = str(PROJECT_ROOT / "results/gaf_convae_model.pth")
    os.makedirs(output_dir, exist_ok=True)

    if args.mode in ("train", "full"):
        # 训练
        print("[GAF+ConvAE] 生成训练数据...")
        train_gafs = generate_train_gafs(data_dir, train_points,
                                          args.window_size, args.n_downsample)
        print(f"[GAF+ConvAE] 训练 ConvAutoencoder...")
        st = time.time()
        model = train_convae(train_gafs, args.window_size, args.epochs,
                              device=args.device)
        train_time = time.time() - st
        print(f"[GAF+ConvAE] 训练完成: {train_time:.1f}s")
        torch.save(model.state_dict(), model_save_path)
        print(f"[GAF+ConvAE] 模型保存: {model_save_path}")
    else:
        # 加载已有模型
        model = ConvAutoencoder(args.window_size).to(args.device)
        model.load_state_dict(torch.load(args.model_path or model_save_path))

    if args.mode in ("detect", "full"):
        # 在测试集上检测
        import glob

        with open(test_points_file) as f:
            test_points = [l.strip() for l in f if l.strip()]

        print(f"\n[GAF+ConvAE] 检测 {len(test_points)} 个测试点位...")
        success = errors = 0

        for i, pt in enumerate(test_points):
            matches = glob.glob(os.path.join(data_dir,
                        f"global_adtk_hbos_*_{pt}_*_trend_seasonal_resid.csv"))
            if not matches:
                errors += 1
                continue

            try:
                mask, rate = detect_single_point(
                    model, matches[0], pt, args.window_size,
                    args.n_downsample, device=args.device)

                pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(
                    os.path.join(output_dir, f"{pt}.csv"), index=False)

                with open(os.path.join(output_dir, f"{pt}.status.json"), "w") as f:
                    json.dump({"status": "success", "anomaly_rate": rate,
                               "anomaly_count": int(mask.sum()),
                               "total_rows": len(mask)}, f, indent=2)
                success += 1

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(test_points)}] success={success}")

            except Exception as e:
                errors += 1
                print(f"  [{i+1}] {pt} ERROR: {e}")

        # 汇总
        print(f"\n[GAF+ConvAE] 完成: {success}/{len(test_points)} 成功")
        status_files = glob.glob(os.path.join(output_dir, "*.status.json"))
        rates = []
        for sf in status_files:
            d = json.load(open(sf))
            if d.get("status") == "success":
                rates.append(d["anomaly_rate"])
        if rates:
            print(f"  mean_rate={np.mean(rates):.4f}, median={np.median(rates):.4f}")
            print(f"  zero={sum(1 for r in rates if r==0)}, over10%={sum(1 for r in rates if r>0.1)}")


if __name__ == "__main__":
    main()

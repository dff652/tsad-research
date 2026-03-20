#!/usr/bin/env python3
"""
ChatTS 批量适配器 —— 一次加载模型，处理多个点位

ChatTS 单次推理仅 ~12s，100 个点位约 20 分钟。

用法：
    PYTHONNOUSERSITE=1 conda run -n chatts python chatts_batch_adapter.py \
        --input-dir /path/to/adtk_hbos_old \
        --output-dir /path/to/predictions/chatts \
        --points-file /path/to/evaluated_points.txt \
        --model-path /home/share/llm_models/bytedance-research/ChatTS-14B \
        --resume --compact
"""

import sys
import os
import time
import json
import glob
import re
import argparse
import numpy as np
import pandas as pd

PLATFORM_INFERENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
sys.path.insert(0, PLATFORM_INFERENCE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--points-file", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-downsample", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prompt-template", default="industrial")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def find_csv(input_dir, point_name):
    matches = glob.glob(os.path.join(input_dir, f"global_adtk_hbos_*_{point_name}_*_trend_seasonal_resid.csv"))
    return matches[0] if matches else None


def get_all_points(input_dir):
    names = []
    for f in sorted(glob.glob(os.path.join(input_dir, "*.csv"))):
        m = re.match(r"global_adtk_hbos_m4_0\.1_1200\.0_(.+)_20230718_trend_seasonal_resid\.csv", os.path.basename(f))
        if m:
            names.append(m.group(1))
    return names


def patch_extract_anomalies():
    """增强 ChatTS 的异常解析鲁棒性"""
    import chatts_detect as _cd
    _original = _cd.extract_anomalies

    def _robust(text):
        try:
            return _original(text)
        except Exception:
            pass
        try:
            match = re.search(r'anomalies\s*=\s*\[(.*?)\]', text, re.DOTALL)
            if not match:
                return []
            content = match.group(1)
            content = content.replace('）', '}').replace('（', '{').replace("'", '"')
            results = []
            for obj_match in re.finditer(r'\{[^{}]+\}', content):
                try:
                    obj = json.loads(obj_match.group())
                    if "range" in obj:
                        results.append(obj)
                except:
                    fixed = re.sub(r',\s*}', '}', obj_match.group())
                    try:
                        obj = json.loads(fixed)
                        if "range" in obj:
                            results.append(obj)
                    except:
                        pass
            return results
        except:
            return []

    _cd.extract_anomalies = _robust


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取点位
    if args.points_file:
        with open(args.points_file) as f:
            points = [l.strip() for l in f if l.strip()]
    else:
        points = get_all_points(args.input_dir)

    # 断点续跑
    if args.resume:
        existing = {os.path.splitext(f)[0] for f in os.listdir(args.output_dir) if f.endswith(".csv")}
        before = len(points)
        points = [p for p in points if p not in existing]
        print(f"断点续跑: 跳过 {before - len(points)}, 剩余 {len(points)}")

    print(f"ChatTS 批量推理: {len(points)} 点位, 模型: {args.model_path}")

    # Patch 解析
    patch_extract_anomalies()

    # 加载模型（一次性）
    from chatts_detect import get_analyzer, chatts_detect
    analyzer = get_analyzer(args.model_path, args.device, load_in_4bit=True,
                             lora_adapter_path=args.lora_path)

    success = errors = 0
    total_time = 0

    for i, point in enumerate(points):
        csv_path = find_csv(args.input_dir, point)
        if not csv_path:
            errors += 1
            continue

        try:
            st = time.time()
            df = pd.read_csv(csv_path)
            skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                    "global_mask_cluster", "local_mask_cluster"}
            value_col = [c for c in df.columns if c not in skip][0]

            data_df = df[[value_col]].copy()
            global_mask, anomalies, _ = chatts_detect(
                data=data_df,
                model_path=args.model_path,
                device=args.device,
                n_downsample=args.n_downsample,
                max_new_tokens=args.max_new_tokens,
                prompt_template_name=args.prompt_template,
                load_in_4bit=True,
                lora_adapter_path=args.lora_path,
            )

            # 保存
            output_path = os.path.join(args.output_dir, f"{point}.csv")
            if args.compact:
                pd.DataFrame({"global_mask": global_mask.astype(np.int8)}).to_csv(output_path, index=False)
            else:
                pd.DataFrame({
                    "Time": df["Time"], "value": df[value_col],
                    "global_mask": global_mask.astype(int),
                }).to_csv(output_path, index=False)

            elapsed = time.time() - st
            total_time += elapsed
            rate = float(global_mask.mean())

            # 状态
            with open(os.path.join(args.output_dir, f"{point}.status.json"), "w") as f:
                json.dump({
                    "status": "success", "anomaly_rate": rate,
                    "anomaly_count": int(global_mask.sum()),
                    "num_intervals": len(anomalies),
                    "total_rows": len(df), "elapsed": elapsed,
                }, f, indent=2)

            success += 1
            print(f"[{i+1}/{len(points)}] {point} - rate={rate:.4f}, intervals={len(anomalies)}, {elapsed:.1f}s")

        except Exception as e:
            errors += 1
            print(f"[{i+1}/{len(points)}] {point} - ERROR: {e}")
            with open(os.path.join(args.output_dir, f"{point}.status.json"), "w") as f:
                json.dump({"status": "error", "error": str(e)}, f)

    print(f"\n{'='*60}")
    print(f"ChatTS 完成: {success}/{len(points)} 成功, {errors} 失败")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    if success > 0:
        print(f"平均每点: {total_time/success:.1f} 秒")


if __name__ == "__main__":
    main()

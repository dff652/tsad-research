#!/usr/bin/env python3
"""
ChatTS 适配器 —— 在 chatts conda 环境中运行

ChatTS-14B 是字节跳动的时序大模型，通过 LLM 理解时序数据进行异常检测。
支持 4-bit 量化，约需 10GB+ 显存。

用法：
    PYTHONNOUSERSITE=1 conda run -n chatts python chatts_adapter.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv \
        --point-name FI6101.PV \
        --model-path /home/share/llm_models/bytedance-research/ChatTS-14B
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd

# 业务平台路径
PLATFORM_INFERENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
sys.path.insert(0, PLATFORM_INFERENCE)


def parse_args():
    parser = argparse.ArgumentParser(description="ChatTS Anomaly Detection Adapter")
    parser.add_argument("--input", required=True, help="输入 CSV")
    parser.add_argument("--output", required=True, help="输出标准格式 CSV")
    parser.add_argument("--point-name", default="unknown")
    parser.add_argument("--model-path", default="/home/share/llm_models/bytedance-research/ChatTS-14B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-downsample", type=int, default=768, help="降采样点数（ChatTS 建议 768）")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prompt-template", default="industrial", help="prompt 模板名")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--lora-path", default=None, help="LoRA 微调适配器路径")
    parser.add_argument("--compact", action="store_true", help="紧凑输出：仅 mask 列")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # 加载数据
        df = pd.read_csv(args.input)
        if "Time" not in df.columns and "time" in df.columns:
            df.rename(columns={"time": "Time"}, inplace=True)

        skip = {"Time", "global_mask", "outlier_mask", "local_mask",
                "global_mask_cluster", "local_mask_cluster"}
        value_col = [c for c in df.columns if c not in skip][0]
        print(f"[ChatTS-Adapter] 点位: {args.point_name}, 数据量: {len(df)}, 值列: {value_col}")

        # 准备输入
        data_df = df[[value_col]].copy()

        # Monkey-patch ChatTS 的 extract_anomalies 使其更鲁棒
        import chatts_detect as _cd
        _original_extract = _cd.extract_anomalies

        def _robust_extract(text):
            """增强版异常解析：容忍常见 LLM JSON 输出错误"""
            import re, json
            # 先尝试原始解析
            try:
                return _original_extract(text)
            except Exception:
                pass
            # 容错解析：修复常见问题
            try:
                # 提取 anomalies = [...] 部分
                match = re.search(r'anomalies\s*=\s*\[(.*?)\]', text, re.DOTALL)
                if not match:
                    return []
                content = match.group(1)
                # 修复中文括号、单引号等
                content = content.replace('）', '}').replace('（', '{')
                content = content.replace("'", '"')
                # 尝试逐个解析对象
                results = []
                for obj_match in re.finditer(r'\{[^{}]+\}', content):
                    try:
                        obj = json.loads(obj_match.group())
                        if "range" in obj:
                            results.append(obj)
                    except json.JSONDecodeError:
                        # 尝试修复后再解析
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

        _cd.extract_anomalies = _robust_extract

        # 调用 ChatTS 检测
        from chatts_detect import chatts_detect

        global_mask, anomalies, position_index = chatts_detect(
            data=data_df,
            model_path=args.model_path,
            device=args.device,
            n_downsample=args.n_downsample,
            max_new_tokens=args.max_new_tokens,
            prompt_template_name=args.prompt_template,
            load_in_4bit=args.load_in_4bit,
            lora_adapter_path=args.lora_path,
        )

        # 保存结果
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.compact:
            result = pd.DataFrame({"global_mask": global_mask.astype(np.int8)})
        else:
            result = pd.DataFrame({
                "Time": df["Time"],
                "value": df[value_col],
                "global_mask": global_mask.astype(int),
                "outlier_mask": global_mask.astype(int),
                "local_mask": np.zeros(len(df), dtype=int),
            })
        result.to_csv(args.output, index=False)

        anomaly_rate = float(global_mask.mean())
        print(f"[ChatTS-Adapter] 完成: rate={anomaly_rate:.6f}, intervals={len(anomalies)}")

        # 状态文件
        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({
                "status": "success",
                "anomaly_rate": anomaly_rate,
                "anomaly_count": int(global_mask.sum()),
                "num_intervals": len(anomalies),
                "total_rows": len(df),
                "model_path": args.model_path,
            }, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        status_path = args.output.replace(".csv", ".status.json")
        with open(status_path, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

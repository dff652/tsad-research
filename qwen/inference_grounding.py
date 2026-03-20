#!/usr/bin/env python3
"""
Qwen VL 系列 Grounding 推理脚本

支持模型：
- Qwen3-VL-8B-Instruct（本地）
- Qwen3.5-27B（本地，多模态）
- 微调后的 LoRA 模型

对测试集图片进行零样本/微调后的异常区域检测。
输出 bbox 坐标，映射回时序索引，生成标准 mask。
"""

import os
import sys
import json
import time
import argparse
import glob
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--images-dir", default=str(PROJECT_ROOT / "qwen/images"))
    parser.add_argument("--annotations-file", default=str(PROJECT_ROOT / "qwen/dataset/annotations_test.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:1", help="GPU 设备")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-points", type=int, default=0, help="0=全部")
    parser.add_argument("--lora-path", default=None, help="LoRA 适配器路径")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def load_model(model_path: str, device: str, lora_path: str = None,
               load_in_4bit: bool = False):
    """加载 Qwen VL 模型"""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[QwenVL] 加载模型: {model_path}")
    print(f"[QwenVL] 设备: {device}, 4bit: {load_in_4bit}")

    processor = AutoProcessor.from_pretrained(model_path)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map=device, trust_remote_code=True,
        )

    if lora_path:
        from peft import PeftModel
        print(f"[QwenVL] 加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    print("[QwenVL] 模型加载完成")
    return model, processor


def run_grounding_inference(model, processor, image_path: str,
                             max_new_tokens: int = 4096) -> str:
    """对单张图片进行异常区域 grounding 推理"""
    import torch

    prompt = (
        'This is a time series plot from an industrial sensor. '
        'Locate every anomaly region in this image. '
        'For each anomaly, report bbox coordinates and anomaly type in JSON format like: '
        '{"bbox_2d": [x1, y1, x2, y2], "label": "anomaly_type"}\n'
        'If no anomaly is found, output: []'
    )

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text


def parse_bbox_response(text: str) -> list:
    """解析模型输出的 bbox JSON"""
    import re as _re

    # 清理 markdown
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0] if text.count("```") >= 2 else text

    # 尝试整体解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if "bbox_2d" in d]
        return []
    except:
        pass

    # 逐个对象解析
    results = []
    for match in _re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(match.group())
            if "bbox_2d" in obj:
                results.append(obj)
        except:
            pass

    return results


def bbox_to_mask(bboxes: list, img_width: int, img_height: int,
                  data_len: int) -> np.ndarray:
    """将 bbox_2d (0-1000 坐标) 转换为时序 mask"""
    mask = np.zeros(data_len, dtype=int)

    for bbox_item in bboxes:
        bbox = bbox_item.get("bbox_2d", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        # x 坐标映射到数据索引（Qwen 0-1000 坐标）
        idx_start = int(x1 / 1000 * data_len)
        idx_end = int(x2 / 1000 * data_len)
        idx_start = max(0, min(idx_start, data_len - 1))
        idx_end = max(idx_start, min(idx_end, data_len))
        mask[idx_start:idx_end] = 1

    return mask


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载标注（用于获取图片列表和元信息）
    with open(args.annotations_file) as f:
        annotations = json.load(f)

    if args.max_points > 0:
        annotations = annotations[:args.max_points]

    print(f"[QwenVL] 推理: {len(annotations)} 张图片, 模型: {args.model_path}")

    # 加载模型
    model, processor = load_model(
        args.model_path, args.device,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
    )

    results = []
    success = errors = 0

    for i, ann in enumerate(annotations):
        point_name = ann["point_name"]
        img_path = os.path.join(args.images_dir, f"{point_name}.png")

        if not os.path.exists(img_path):
            errors += 1
            continue

        try:
            st = time.time()
            output_text = run_grounding_inference(model, processor, img_path, args.max_new_tokens)
            elapsed = time.time() - st

            # 解析 bbox
            bboxes = parse_bbox_response(output_text)

            # 转换为 mask（使用降采样后的长度）
            data_len = ann["downsampled_length"]
            mask = bbox_to_mask(bboxes, ann["image_size"][0], ann["image_size"][1], data_len)

            # 保存
            pred_path = os.path.join(args.output_dir, f"{point_name}.csv")
            pd.DataFrame({"global_mask": mask.astype(np.int8)}).to_csv(pred_path, index=False)

            rate = float(mask.mean())
            result = {
                "point_name": point_name,
                "status": "success",
                "anomaly_rate": rate,
                "num_bboxes": len(bboxes),
                "num_gt_anomalies": ann["num_anomalies"],
                "elapsed": elapsed,
                "bboxes": bboxes,
                "raw_output": output_text[:500],
            }
            results.append(result)

            with open(os.path.join(args.output_dir, f"{point_name}.status.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            success += 1
            print(f"[{i+1}/{len(annotations)}] {point_name}: "
                  f"rate={rate:.4f}, bboxes={len(bboxes)}, gt={ann['num_anomalies']}, "
                  f"{elapsed:.1f}s")

        except Exception as e:
            errors += 1
            print(f"[{i+1}/{len(annotations)}] {point_name}: ERROR - {e}")
            results.append({"point_name": point_name, "status": "error", "error": str(e)})

        # 定期清理 GPU 缓存
        if (i + 1) % 10 == 0:
            import torch
            torch.cuda.empty_cache()

    # 汇总
    ok_results = [r for r in results if r.get("status") == "success"]
    print(f"\n{'='*60}")
    print(f"完成: {success}/{len(annotations)}, 错误: {errors}")
    if ok_results:
        rates = [r["anomaly_rate"] for r in ok_results]
        bbox_counts = [r["num_bboxes"] for r in ok_results]
        print(f"  mean_rate={np.mean(rates):.4f}, mean_bboxes={np.mean(bbox_counts):.1f}")
        print(f"  zero_detection={sum(1 for b in bbox_counts if b == 0)}")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"results": results, "success": success, "errors": errors}, f,
                  indent=2, ensure_ascii=False)
    print(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()

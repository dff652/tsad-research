#!/usr/bin/env python3
"""
Qwen3-VL-8B Grounding LoRA 微调

用 337 个训练点位的异常 bbox 标注，微调 Qwen3-VL-8B 的异常区域定位能力。

LoRA 微调策略（节省显存）：
- rank=16, alpha=32
- 仅微调 attention 层
- 4-bit 量化基座模型
- 批量大小 1，梯度累积 4

用法：
    PYTHONNOUSERSITE=1 conda run -n qwen_tune python finetune_grounding.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 配置
MODEL_PATH = "/home/share/llm_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = str(PROJECT_ROOT / "qwen/lora_output")
TRAIN_DATA = str(PROJECT_ROOT / "qwen/dataset/conversations_train.jsonl")
IMAGES_DIR = str(PROJECT_ROOT / "qwen/images")

LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048
DEVICE = "cuda:1"


class GroundingDataset(Dataset):
    """Grounding 微调数据集"""

    def __init__(self, data_path: str, images_dir: str, processor, max_samples: int = 0):
        self.processor = processor
        self.images_dir = images_dir
        self.samples = []

        with open(data_path) as f:
            for line in f:
                sample = json.loads(line.strip())
                # 检查图片是否存在
                msg = sample["messages"]
                point_name = sample.get("metadata", {}).get("point_name", "")
                img_path = os.path.join(images_dir, f"{point_name}.png")
                if os.path.exists(img_path) and sample["metadata"].get("num_anomalies", 0) > 0:
                    self.samples.append(sample)

        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        print(f"[Dataset] 加载 {len(self.samples)} 个训练样本（有异常标注的）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        point_name = sample["metadata"]["point_name"]
        img_path = os.path.join(self.images_dir, f"{point_name}.png")

        # 构造消息
        user_msg = sample["messages"][0]
        assistant_msg = sample["messages"][1]

        # 加载图片
        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "user_text": user_msg["content"][1]["text"],
            "assistant_text": assistant_msg["content"],
            "point_name": point_name,
        }


def main():
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Finetune] 加载模型: {MODEL_PATH}")
    print(f"[Finetune] 输出目录: {OUTPUT_DIR}")

    # 加载 processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 4-bit 量化加载基座
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        device_map=DEVICE, trust_remote_code=True,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据集
    dataset = GroundingDataset(TRAIN_DATA, IMAGES_DIR, processor)

    # 训练循环
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    model.train()
    total_steps = 0
    total_loss = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_steps = 0

        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample["image"]
            user_text = sample["user_text"]
            target_text = sample["assistant_text"]

            # 构造完整对话
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]

            text = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # 设置 labels（causal LM，只对 assistant 回复计算 loss）
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            # 简化：对整个序列计算 loss（包含 assistant 部分）
            inputs["labels"] = labels

            try:
                outputs = model(**inputs)
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()

                epoch_loss += loss.item() * GRAD_ACCUM
                epoch_steps += 1
                total_steps += 1

                if total_steps % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                if total_steps % 50 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Step {epoch_steps}, "
                          f"Loss: {avg_loss:.4f}")

            except Exception as e:
                print(f"  Skip sample {sample['point_name']}: {e}")
                optimizer.zero_grad()
                continue

            # 定期清理
            if total_steps % 20 == 0:
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} 完成: avg_loss={avg_epoch_loss:.4f}, steps={epoch_steps}")

        # 每个 epoch 保存 checkpoint
        ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch+1}")
        model.save_pretrained(ckpt_dir)
        print(f"  Checkpoint 保存: {ckpt_dir}")

    # 保存最终 LoRA 权重
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    print(f"\n微调完成！LoRA 权重: {final_dir}")

    # 保存训练配置
    config = {
        "model_path": MODEL_PATH,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "train_samples": len(dataset),
        "total_steps": total_steps,
    }
    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()

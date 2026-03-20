"""
@File    :   chatts_detect.py
@Time    :   2024/12/05
@Author  :   DouFengfeng
@Desc    :   基于 ChatTS 大模型的时序异常检测

ChatTS-14B 是字节跳动开源的时序分析模型，本模块将其封装为可在 run.py 中调用的异常检测方法。
注意：该方法需要 GPU 资源（4-bit 量化约需 10GB+ 显存）。
"""

import math
import os
from typing import List, Tuple, Optional, Dict, Any
import re
import json
import time
from benchmark.lib import patch_transformers # Monkey Patch for transformers 4.54+

# Prompt模板缓存
_prompt_templates_cache: Optional[Dict] = None


def load_prompt_templates(prompts_file: str = None) -> Dict:
    """
    加载prompt模板配置文件
    
    Args:
        prompts_file: prompt配置文件路径，默认为configs/chatts_prompts.json
        
    Returns:
        prompt模板字典
    """
    global _prompt_templates_cache
    
    if _prompt_templates_cache is not None:
        return _prompt_templates_cache
    
    if prompts_file is None:
        # 默认路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_file = os.path.join(script_dir, 'configs', 'chatts_prompts.json')
    
    if os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            _prompt_templates_cache = json.load(f)
        print(f"[ChatTS] 已加载 {len(_prompt_templates_cache)} 个prompt模板")
    else:
        print(f"[ChatTS] 未找到prompt配置文件: {prompts_file}，使用内置默认模板")
        _prompt_templates_cache = {}
    
    return _prompt_templates_cache


def get_prompt_template(template_name: str = "default") -> str:
    """
    获取指定名称的prompt模板
    
    Args:
        template_name: 模板名称，如 "default", "detailed", "minimal", "industrial", "english"
        
    Returns:
        prompt模板字符串
    """
    templates = load_prompt_templates()
    
    if template_name in templates:
        template = templates[template_name].get("template", "")
        print(f"[ChatTS] 使用prompt模板: {template_name} - {templates[template_name].get('name', '')}")
        return template
    else:
        print(f"[ChatTS] 未找到模板 '{template_name}'，使用内置默认模板")
        return None


def list_prompt_templates() -> List[Dict]:
    """
    列出所有可用的prompt模板
    
    Returns:
        模板信息列表
    """
    templates = load_prompt_templates()
    result = []
    for key, val in templates.items():
        result.append({
            "id": key,
            "name": val.get("name", key),
            "description": val.get("description", "")
        })
    return result

import numpy as np
import pandas as pd

# GPU 相关依赖（懒加载）
_torch = None
_ChatTSAnalyzer = None


def _lazy_import_torch():
    """懒加载 torch，避免无 GPU 环境下报错"""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def extract_anomalies(text: str) -> List[Dict]:
    """
    从包含 'anomalies = [...]' 的字符串中提取并解析 anomalies 数组。
    如果数组被截断，将尽力截到最后一个完整对象并补全右方括号。
    
    Args:
        text: 模型输出的文本
        
    Returns:
        异常列表，每个元素包含 range, amp, label, detail, color 等字段
    """
    # 1) 找到 'anomalies = [' 的起点
    m = re.search(r'anomalies\s*=\s*\[', text)
    if not m:
        raise ValueError("No 'anomalies = [' found in text")
    start = m.end() - 1  # 指向 '['

    # 2) 逐字符扫描，找到匹配的 ']'；期间忽略字符串内的括号
    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break

    raw = text[start:end+1] if end is not None else text[start:]  # 可能截断

    # 3) 如果截断：裁到最后一个完整的 '}' 并补 ']'
    if end is None:
        last_closing_brace = raw.rfind('}')
        if last_closing_brace == -1:
            raise ValueError("Found anomalies '[', but no complete object to parse.")
        raw = raw[:last_closing_brace+1] + ']'

    # 4) 解析为 JSON
    try:
        anomalies = json.loads(raw)
    except json.JSONDecodeError:
        # 常见修复：去掉尾随逗号
        raw_fixed = re.sub(r',\s*]', ']', raw)
        anomalies = json.loads(raw_fixed)

    if not isinstance(anomalies, list):
        raise ValueError("Parsed anomalies is not a list.")
    return anomalies


def map_anomalies_to_original(
    anomalies: List[Dict], 
    position_index: np.ndarray
) -> List[Dict]:
    """
    将下采样序列上的异常索引映射到原始 DataFrame 索引。
    
    Args:
        anomalies: 模型返回的异常列表，每个元素包含 "range": [start, end]
        position_index: ts_downsample 返回的 position_index（整数位置索引数组）
    
    Returns:
        映射后的异常列表，range 变为原始数据的位置索引
    """
    idx_array = np.asarray(position_index)
    
    mapped = []
    print(f"[ChatTS][Debug] map_anomalies_to_original received {len(anomalies)} items.")
    for i, a in enumerate(anomalies):
        try:
            if "range" not in a:
                print(f"[ChatTS][Warning] Skipping anomaly without 'range' field: {a}")
                continue
                
            ds_start, ds_end = a["range"]
            
            # 边界检查
            ds_start = max(0, min(ds_start, len(idx_array) - 1))
            ds_end = max(0, min(ds_end, len(idx_array) - 1))
            
            # 通过 position_index 映射到原始位置索引
            orig_start = int(idx_array[ds_start])
            orig_end = int(idx_array[ds_end])
            
            mapped_anomaly = a.copy()
            mapped_anomaly["range"] = [orig_start, orig_end]
            mapped_anomaly["downsampled_range"] = [a["range"][0], a["range"][1]]
            mapped.append(mapped_anomaly)
        except KeyError as e:
            print(f"[ChatTS][Error] KeyError anomaly[{i}] type={type(a)} keys={list(a.keys()) if isinstance(a, dict) else 'N/A'}: {e}")
            continue
        except Exception as e:
            print(f"[ChatTS][Error] Unexpected error anomaly[{i}]: {e}")
            continue
    
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
    print(f"[ChatTS][Debug] create_mask_from_anomalies received {len(anomalies)} items.")
    for i, a in enumerate(anomalies):
        try:
            if "range" not in a:
                continue
            start, end = a["range"]
            start = max(0, min(start, data_length - 1))
            end = max(0, min(end, data_length - 1))
            mask[start:end+1] = 1
        except KeyError as e:
             print(f"[ChatTS][Error] KeyError mask creation anomaly[{i}]: {e}")
             continue
        except Exception as e:
            print(f"[ChatTS][Error] Unexpected error in mask creation for anomaly[{i}] {a}: {e}")
            continue
    return mask


class ChatTSAnalyzer:
    """
    ChatTS-14B 时序分析器（推理版）
    - 单卡推理（推荐 4-bit 量化以控制显存）
    - 支持长序列按滑窗推理并合并
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        attn_implementation: str = "eager",
        torch_dtype=None,
        lora_adapter_path: str = None,
    ):
        """
        Args:
            model_path: 本地或HF路径
            device: 统一放到同一张卡上，例如 "cuda:0"
            load_in_4bit: 是否使用 bitsandbytes 4-bit 量化
            attn_implementation: 'eager' / 'sdpa' / 'flash_attention_2'
            torch_dtype: 建议 fp16 或 bfloat16
            lora_adapter_path: LoRA 微调适配器路径（可选）
        """
        torch = _lazy_import_torch()
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
            
        self.model_path = model_path
        self.compute_dtype = torch_dtype
        self.lora_adapter_path = lora_adapter_path
        
        # 解析设备配置，支持多种格式：
        # - "auto": 自动分布到所有可用 GPU
        # - "cuda:0": 单 GPU
        # - "cuda:0,cuda:1" 或 "0,1": 手动指定多 GPU
        self._max_memory = None  # 初始化，可能在 _parse_device_config 中设置
        device_map = self._parse_device_config(device)
        self._device_config = device  # 保存原始配置用于日志
        
        bnb_config = None
        
        # 检测是否是 ChatTS-8B 模型（基于 Qwen3，对量化敏感）
        is_8b_model = '8B' in model_path or '8b' in model_path
        if is_8b_model and load_in_4bit:
            print("[ChatTS] ⚠️ 检测到 ChatTS-8B 模型，4-bit 量化会严重影响输出质量！")
            print("[ChatTS] ⚠️ 自动禁用 4-bit 量化以确保正常输出（需要约 16GB 显存）")
            load_in_4bit = False
        
        # ChatTS-8B 使用 eager attention 会导致输出乱码，需要使用默认 SDPA
        if is_8b_model and attn_implementation == 'eager':
            print("[ChatTS] ⚠️ ChatTS-8B 与 eager attention 不兼容，切换为默认 SDPA")
            attn_implementation = None  # 使用默认值
        
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "quantization_config": bnb_config,
                "low_cpu_mem_usage": True,
                "device_map": device_map,
            }
            # 如果指定了 max_memory（手动多 GPU 模式），添加到参数中
            if self._max_memory is not None:
                model_kwargs["max_memory"] = self._max_memory
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation
            
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "device_map": device_map,
            }
            # 如果指定了 max_memory（手动多 GPU 模式），添加到参数中
            if self._max_memory is not None:
                model_kwargs["max_memory"] = self._max_memory
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation
            
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # 加载 LoRA 适配器（如果提供）
        if lora_adapter_path is not None:
            from peft import PeftModel
            print(f"[ChatTS] 正在加载 LoRA 适配器: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                lora_adapter_path,
                is_trainable=False  # 推理模式
            )
            print("[ChatTS] LoRA 适配器加载完成")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, tokenizer=self.tokenizer
        )

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # 获取模型输入所在的设备（多 GPU 时为第一个设备）
        self.device = self._get_model_input_device()

    def _build_prompt(self, timeseries_len: int, system_prompt: str, task_prompt_tpl: str) -> str:
        # 使用 replace 而不是 format，避免 JSON 示例中的花括号被误认为是占位符
        # 仅替换 {ts_len}，不影响其他内容
        if "{ts_len}" in task_prompt_tpl:
            user_prompt = task_prompt_tpl.replace("{ts_len}", str(timeseries_len))
        else:
            # 兼容可能的旧模板（虽然不推荐，但防止 crash）
            user_prompt = task_prompt_tpl
            
        # 统一占位符为 <ts><ts/>
        if "<ts><ts/>" not in user_prompt:
            # 兼容老写法 <ts></ts>
            if "<ts></ts>" in user_prompt:
                user_prompt = user_prompt.replace("<ts></ts>", "<ts><ts/>")
            else:
                # 若模板中没有任何占位符，强制追加一段
                user_prompt = user_prompt.rstrip() + "\n下面是时间序列：<ts><ts/>"
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>"
            f"<|im_start|>user\n{user_prompt}<|im_end|><|im_start|>assistant\n"
        )
        return prompt

    def _parse_device_config(self, device: str):
        """
        解析设备配置字符串，返回 device_map 参数
        
        支持格式：
        - "auto": 自动分布到所有可用 GPU
        - "balanced": 均匀分布到所有可用 GPU
        - "cuda:0": 单 GPU（设备索引 0）
        - "cuda:0,cuda:1" 或 "0,1": 手动指定多 GPU
        
        Returns:
            device_map 参数，可以是字符串、整数或列表
        """
        device = device.strip()
        
        # 自动模式
        if device.lower() in ("auto", "balanced"):
            print(f"[ChatTS] 使用 {device} 模式分布模型到多 GPU")
            return device.lower()
        
        # 检查是否包含逗号（多 GPU 手动指定）
        if "," in device:
            # 解析多 GPU 配置，如 "cuda:0,cuda:1" 或 "0,1"
            gpu_ids = []
            for part in device.split(","):
                part = part.strip()
                if part.startswith("cuda:"):
                    gpu_ids.append(int(part.split(":")[1]))
                elif part.isdigit():
                    gpu_ids.append(int(part))
            
            if len(gpu_ids) > 1:
                # 使用 accelerate 的 max_memory 参数分配到指定 GPU
                torch = _lazy_import_torch()
                max_memory = {}
                for gpu_id in gpu_ids:
                    # 获取每个 GPU 的可用显存
                    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
                        # 预留 1GB 给系统
                        max_memory[gpu_id] = f"{int(total_mem / 1024**3) - 1}GiB"
                max_memory["cpu"] = "32GiB"  # CPU 备用内存
                print(f"[ChatTS] 使用手动指定的多 GPU: {gpu_ids}，max_memory: {max_memory}")
                # 返回 auto，但会通过 max_memory 限制使用的 GPU
                self._max_memory = max_memory
                return "auto"
            elif len(gpu_ids) == 1:
                print(f"[ChatTS] 使用单 GPU: cuda:{gpu_ids[0]}")
                return {"": gpu_ids[0]}
        
        # 单 GPU 模式，如 "cuda:0"
        m = re.match(r"cuda:(\d+)", device)
        if m:
            device_index = int(m.group(1))
            print(f"[ChatTS] 使用单 GPU: cuda:{device_index}")
            return {"": device_index}
        
        # 默认：尝试作为设备字符串
        print(f"[ChatTS] 使用设备: {device}")
        return device
    
    def _get_model_input_device(self):
        """
        获取模型输入张量应该放置的设备
        对于多 GPU 模型，返回第一层所在的设备
        """
        torch = _lazy_import_torch()
        
        # 尝试获取模型的 hf_device_map（accelerate 分布后会有这个属性）
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            # 找到第一个层的设备
            first_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(first_device, int):
                return torch.device(f"cuda:{first_device}")
            return torch.device(first_device)
        
        # 如果是 PEFT 模型，尝试获取 base_model
        model = self.model
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model"):
            model = model.model
        
        # 尝试从第一个参数获取设备
        try:
            first_param = next(model.parameters())
            return first_param.device
        except StopIteration:
            pass
        
        # 默认使用 cuda:0
        return torch.device("cuda:0")
        
    def _prepare_inputs(self, prompt: str, timeseries: np.ndarray):
        torch = _lazy_import_torch()
        # 关键断言和调试：确保 prompt 中包含占位符
        has_placeholder = "<ts><ts/>" in prompt
        if not has_placeholder:
            print("[ChatTS][警告] Prompt 中未找到 <ts><ts/> 占位符，将无法绑定时间序列！前200字符如下：")
            print(prompt[:200].replace("\n", "\\n"))
        if timeseries is None or len(timeseries) == 0:
            print("[ChatTS][警告] 传入的时间序列为空！")
        inputs = self.processor(
            text=[prompt],
            timeseries=[timeseries],
            padding=True,
            return_tensors="pt",
        )
        model_dtype = self.compute_dtype
        # 使用动态获取的输入设备
        input_device = self.device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.to(input_device)
                if v.is_floating_point():
                    v = v.to(model_dtype)
                inputs[k] = v
        return inputs

    def _generate(self, inputs, max_new_tokens: int = 1024, top_p: float = 0.9, use_cache: Optional[bool] = None) -> str:
        torch = _lazy_import_torch()
        
        # 检测模型架构类型
        model_type = getattr(self.model.config, 'model_type', '').lower()
        # 放宽检测逻辑：只要是 Qwen 系列或 ChatTS，都尝试应用补丁
        is_qwen = 'qwen' in model_type or 'qwen' in self.model_path.lower() or 'chatts' in self.model_path.lower()
        
        # 检测是否使用多 GPU（通过 hf_device_map 判断）
        is_multi_gpu = False
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            devices = set(self.model.hf_device_map.values())
            # 如果有多个不同的设备，说明是多 GPU
            is_multi_gpu = len(devices) > 1
        
        # Qwen3 在处理时序嵌入时，KV cache 与 attention mask 存在兼容性问题
        # 多 GPU 模式下，DynamicCache.seen_tokens 与 transformers 4.57.3 不兼容，需禁用 cache
        if use_cache is not None:
            pass
        elif is_multi_gpu:
            # 多 GPU 模式下禁用 cache 以避免兼容性问题
            print("[ChatTS] 检测到多 GPU 模式，禁用 KV cache 以避免兼容性问题")
            use_cache = False
        else:
            # 默认启用 Cache，依靠 Monkey Patch 修复
            use_cache = True
        
        gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=use_cache)
        
        # Monkey Patch for Qwen3 dimension mismatch
        # 递归遍历 PEFT/LoRA 包装结构，找到包含 _update_causal_mask 的最内层模型实例
        import types
        
        def find_update_causal_mask_target(model, depth=0):
            """递归查找包含 _update_causal_mask 方法的模型实例
            
            PEFT 模型结构示例:
            PeftModelForCausalLM -> LoraModel -> Qwen3TSForCausalLM -> Qwen3Model
            需要遍历到最内层的 Qwen3Model 才能找到 _update_causal_mask
            """
            if depth > 10:  # 防止无限递归
                return None
            
            # 检查当前层是否有目标方法
            if hasattr(model, "_update_causal_mask"):
                return model
            
            # PEFT 包装：尝试 base_model 属性
            if hasattr(model, "base_model"):
                result = find_update_causal_mask_target(model.base_model, depth + 1)
                if result is not None:
                    return result
            
            # Transformer 模型：尝试 model 属性（如 Qwen3TSForCausalLM.model -> Qwen3Model）
            if hasattr(model, "model") and model.model is not model:
                result = find_update_causal_mask_target(model.model, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        target_instance = find_update_causal_mask_target(self.model)
        
        # 应用 Instance-level Patch
        if target_instance:
            if not hasattr(target_instance, "_is_patched_for_ts"):
                print(f"[ChatTS] 检测到 {target_instance.__class__.__name__} 包含 _update_causal_mask，正在应用 Monkey Patch...")
                
                # 保存原方法
                target_instance._original_update_causal_mask = target_instance._update_causal_mask
                
                def _patched_update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
                    # 如果 input_tensor (Embedding后) 长度大于 cache_position (Input IDs后) 长度
                    # 说明发生了时序注入，需要重新生成 cache_position
                    if input_tensor.shape[1] > cache_position.shape[0]:
                        # print(f"[Debug] 修正 cache_position: {cache_position.shape[0]} -> {input_tensor.shape[1]}")
                        cache_position = torch.arange(
                            input_tensor.shape[1], device=input_tensor.device
                        )
                    
                    return self._original_update_causal_mask(
                        attention_mask, input_tensor, cache_position, past_key_values, output_attentions
                    )
                
                # 绑定新方法到实例
                target_instance._update_causal_mask = types.MethodType(_patched_update_causal_mask, target_instance)
                target_instance._is_patched_for_ts = True
                print(f"[ChatTS] Monkey Patch 应用成功")
            else:
                 # 已 Patch 过，无需重复打印
                 pass

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )
        return text.strip()

    def _run_one_window(
        self, timeseries: np.ndarray, max_new_tokens: int, top_p: float,
        system_prompt: str, task_prompt_tpl: str, use_cache: Optional[bool] = None
    ) -> str:
        prompt = self._build_prompt(
            timeseries_len=len(timeseries),
            system_prompt=system_prompt,
            task_prompt_tpl=task_prompt_tpl,
        )
        inputs = self._prepare_inputs(prompt, timeseries)
        return self._generate(inputs, max_new_tokens=max_new_tokens, top_p=top_p, use_cache=use_cache)

    @staticmethod
    def _make_windows(n: int, window_len: int, overlap: float) -> List[Tuple[int, int]]:
        assert 0 <= overlap < 1, "overlap 需在 [0,1) 之间"
        if n <= window_len:
            return [(0, n)]
        stride = max(1, int(window_len * (1 - overlap)))
        starts = list(range(0, max(1, n - window_len + 1), stride))
        if starts[-1] + window_len < n:
            starts.append(n - window_len)
        return [(s, min(n, s + window_len)) for s in starts]

    def analyze(
        self,
        timeseries: np.ndarray,
        max_new_tokens: int = 1024,
        window_len: Optional[int] = None,
        overlap: float = 0.25,
        per_window_new_tokens: Optional[int] = None,
        top_p: float = 1,
        system_prompt: str = "You are a helpful assistant.",
        task_prompt_tpl: str = None,
        clear_cuda_cache_each_window: bool = False,
        header_each_window: bool = True,
        use_cache: Optional[bool] = None,
    ) -> str:
        """
        分析时序数据
        
        Args:
            timeseries: 一维 numpy 数组
            max_new_tokens: 单窗/整段的最大生成长度
            window_len: 若为 None 则整段推理；否则滑窗
            overlap: 滑窗重叠比例（0~<1）
            top_p: nucleus sampling；=1 时贪心
        """
        torch = _lazy_import_torch()
        
        if task_prompt_tpl is None:
            task_prompt_tpl = self._get_default_prompt()
            
        assert timeseries.ndim == 1, "timeseries 需要是一维数组"

        if window_len is None or len(timeseries) <= window_len:
            return self._run_one_window(
                timeseries=timeseries,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                system_prompt=system_prompt,
                task_prompt_tpl=task_prompt_tpl,
                use_cache=use_cache,
            )

        # 滑窗模式
        windows = self._make_windows(n=len(timeseries), window_len=window_len, overlap=overlap)
        num_windows = len(windows)
        pnt = per_window_new_tokens or max(128, min(1024, max_new_tokens // max(1, num_windows)))

        pieces: List[str] = []
        for i, (s, e) in enumerate(windows, 1):
            seg = timeseries[s:e]
            try:
                txt = self._run_one_window(
                    timeseries=seg, max_new_tokens=pnt, top_p=top_p,
                    system_prompt=system_prompt, task_prompt_tpl=task_prompt_tpl,
                    use_cache=use_cache,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fallback_tokens = max(64, pnt // 2)
                txt = self._run_one_window(
                    timeseries=seg, max_new_tokens=fallback_tokens, top_p=top_p,
                    system_prompt=system_prompt, task_prompt_tpl=task_prompt_tpl,
                    use_cache=use_cache,
                )

            if header_each_window:
                pieces.append(f"[Window {i}/{num_windows}: {s}-{e}]\n{txt}")
            else:
                pieces.append(txt)

            if clear_cuda_cache_each_window:
                torch.cuda.empty_cache()

        return "\n\n".join(pieces)
    
    @staticmethod
    def _get_default_prompt() -> str:
        """默认的异常检测 prompt"""
        return (
            "我有一个长度为 {ts_len} 的时间序列：<ts><ts/>。"
            "请识别该时间序列中所有异常或异常片段。\n\n"
            "从全局视角找出具有统计显著性的异常（如极值、突变），忽略正常周期波动。\n\n"
            "【输出要求】\n"
            "仅输出一个名为 anomalies 的 JSON 数组，不要输出任何其他文字或代码块标记。\n"
            "每个元素必须严格包含以下4个字段：\n"
            "- range: [起始索引, 结束索引]，整数\n"
            "- amp: 异常幅度，保留两位小数\n"
            "- label: 简短标签（不超过10字），如\"向下尖峰\"、\"突增\"、\"趋势漂移\"\n"
            "- detail: 简短描述（不超过30字）\n\n"
            "【格式示例】\n"
            "anomalies = [\n"
            '    {{"range": [137, 139], "amp": 1.91, "label": "向下尖峰", "detail": "数值从1.91骤降至0后恢复"}},\n'
            '    {{"range": [500, 520], "amp": 5.23, "label": "趋势漂移", "detail": "整体水平逐渐上升约5.23"}}\n'
            "]\n\n"
            "注意：必须使用双引号，每个对象的字段顺序为 range、amp、label、detail。"
        )


# 全局模型实例（单例模式，避免重复加载）
_analyzer_instance: Optional[ChatTSAnalyzer] = None
_analyzer_config: Optional[Dict] = None


def get_analyzer(
    model_path: str,
    device: str = "cuda:0",
    load_in_4bit: bool = True,
    lora_adapter_path: str = None,
) -> ChatTSAnalyzer:
    """
    获取 ChatTSAnalyzer 实例（单例模式）
    
    如果配置相同则复用现有实例，否则重新创建
    
    Args:
        model_path: 基础模型路径
        device: GPU 设备
        load_in_4bit: 是否 4-bit 量化
        lora_adapter_path: LoRA 微调适配器路径（可选）
    """
    global _analyzer_instance, _analyzer_config
    
    new_config = {
        "model_path": model_path, 
        "device": device, 
        "load_in_4bit": load_in_4bit,
        "lora_adapter_path": lora_adapter_path,
    }
    
    if _analyzer_instance is None or _analyzer_config != new_config:
        adapter_info = f" + LoRA({lora_adapter_path})" if lora_adapter_path else ""
        print(f"[ChatTS] 正在加载模型: {model_path}{adapter_info} 到 {device}...")
        _analyzer_instance = ChatTSAnalyzer(
            model_path=model_path,
            device=device,
            load_in_4bit=load_in_4bit,
            lora_adapter_path=lora_adapter_path,
        )
        _analyzer_config = new_config
        print("[ChatTS] 模型加载完成")
    
    return _analyzer_instance


def chatts_detect(
    data: pd.DataFrame,
    model_path: str,
    device: str = "cuda:0",
    n_downsample: int = 768,
    downsampler: str = "m4",
    max_new_tokens: int = 2048,
    prompt_template: str = None,
    prompt_template_name: str = "default",
    load_in_4bit: bool = True,
    use_cache: Optional[bool] = None,
    lora_adapter_path: str = None,
) -> Tuple[np.ndarray, List[Dict], Optional[np.ndarray]]:
    """
    使用 ChatTS 进行异常检测
    
    Args:
        data: 输入数据，DataFrame 格式，第一列为时序值
        model_path: ChatTS 基础模型路径
        device: GPU 设备（如 "cuda:0"）
        n_downsample: 降采样点数
        downsampler: 降采样方法（'m4' 或 'minmax'）
        max_new_tokens: 最大生成 token 数
        prompt_template: 自定义 prompt 字符串（可选，优先级最高）
        prompt_template_name: 预设模板名称（如 "default", "detailed", "minimal", "industrial", "english"）
        load_in_4bit: 是否使用 4-bit 量化
        use_cache: 是否使用 KV cache（可选，None 时自动判断）
        lora_adapter_path: LoRA 微调适配器路径（可选）
    
    Returns:
        global_mask: 与原始数据对齐的异常掩码（0=正常，1=异常）
        anomalies: 异常区间详细信息列表（已映射到原始索引）
        position_index: 本次检测所用的降采样位置索引（未降采样时为 np.arange(data_length)）
    """
    from benchmark.lib.signal_utils import ts_downsample
    
    # 处理prompt模板：优先使用自定义字符串，否则从配置加载
    if prompt_template is None and prompt_template_name:
        prompt_template = get_prompt_template(prompt_template_name)
    
    # 获取列名和数据
    column = data.columns[0]
    series = data[column]
    data_length = len(series)
    
    # 降采样
    if downsampler is None or str(downsampler).lower() == "none":
        ts_values = series.values.astype(np.float32)
        position_index = np.arange(data_length)
    elif data_length > n_downsample:
        downsampled_data, _, position_index = ts_downsample(series, downsampler=downsampler, n_out=n_downsample)
        ts_values = downsampled_data.values.astype(np.float32)
    else:
        ts_values = series.values.astype(np.float32)
        position_index = np.arange(data_length)
    
    # 获取分析器实例
    analyzer = get_analyzer(model_path, device, load_in_4bit, lora_adapter_path)
    
    # 执行推理
    st = time.time()
    text = analyzer.analyze(
        ts_values,
        max_new_tokens=max_new_tokens,
        top_p=1,
        task_prompt_tpl=prompt_template,
        use_cache=use_cache,
    )
    et = time.time()
    print(f"[ChatTS] 推理耗时: {et - st:.2f}s")
    
    # 解析结果
    try:
        anomalies = extract_anomalies(text)
        print(f"[ChatTS][DEBUG] Raw anomalies: {anomalies}")
        print(f"[ChatTS] 检测到 {len(anomalies)} 个异常区间")
    except ValueError as e:
        print(f"[ChatTS] 解析结果失败: {e}")
        print(f"[ChatTS][DEBUG] Raw output text: {text[:1000]}") # 打印更多字符
        anomalies = []
    
    # 映射到原始索引
    if anomalies:
        mapped_anomalies = map_anomalies_to_original(anomalies, position_index)
    else:
        mapped_anomalies = []
    
    # 创建掩码
    global_mask = create_mask_from_anomalies(data_length, mapped_anomalies)
    
    return global_mask, mapped_anomalies, position_index

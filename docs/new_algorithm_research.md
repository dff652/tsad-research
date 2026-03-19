# 新算法调研报告

> 更新时间：2026-03-19
> 目标：里程碑 3 —— 新范式沙盒引入，寻找有望突破 Timer SOTA 的前瞻性方案

---

## 一、候选算法评估

### 1.1 MOMENT (CMU, ICML'24) — 推荐引入

- **架构**: T5-based, 385M 参数（Small/Base/Large 三个规格）
- **特点**: 预训练重建头，零样本异常检测（无需微调）
- **原理**: 重建输入时序，残差大的点 = 异常（与 Timer 类似但用 T5 架构）
- **权重**: `AutonLab/MOMENT-1-large` (HuggingFace)
- **优势**: 多任务（预测+分类+异常+补全），已有 anomaly_detection tutorial
- **预计接入难度**: 中等（需新建 conda 环境，HF transformers 兼容）
- **论文**: https://arxiv.org/abs/2402.03885
- **代码**: https://github.com/moment-timeseries-foundation-model/moment

### 1.2 Toto (Datadog, 2025) — 强烈推荐

- **架构**: Transformer-based TSFM，专为可观测性设计
- **特点**: 零样本预测+异常检测，2 万亿数据点预训练
- **权重**: `Datadog/Toto-Open-Base-1.0` (HuggingFace, Apache 2.0)
- **优势**: 在 BOOM 和 GIFT-Eval benchmark 上 SOTA；专为运维/可观测场景优化，与我们的工业传感器场景高度匹配
- **预计接入难度**: 中等
- **代码**: https://github.com/DataDog/toto

### 1.3 TimeRCD (2025) — 关注

- **架构**: 基于 Relative Context Discrepancy 的新预训练范式
- **特点**: 通过检测相邻时间窗口的显著差异来识别异常
- **优势**: 专为 TSAD 设计的预训练方法（不是简单复用预测模型）
- **预计接入难度**: 高（可能需要自行训练）

### 1.4 TSB-AD Benchmark (2025) — 参考

- **内容**: 1070 条高质量时序 + 40 种检测算法评测
- **价值**: 可参考其评测方法和指标设计
- **关键发现**: 简单统计方法在很多场景下不输于复杂神经网络
- **代码**: https://github.com/TheDatumOrg/TSB-AD

---

## 二、引入优先级

| 优先级 | 算法 | 理由 | 预计耗时 |
|--------|------|------|---------|
| **P0** | MOMENT-1-large | 零样本异常检测，HF 直接可用，论文引用高 | 1-2 天 |
| **P0** | Toto-Open | Datadog 出品 SOTA，工业场景高度匹配 | 1-2 天 |
| P1 | ChatTS LoRA 微调版 | 已有 LoRA 权重，可能显著提升 ChatTS 表现 | 0.5 天 |
| P2 | TimeRCD | 需要更深入调研，可能需要训练 | 3-5 天 |

---

## 三、接入方案（遵循红线规则）

### 环境隔离
```
conda create -n moment python=3.10 pytorch transformers  # MOMENT 专属
conda create -n toto python=3.10 pytorch               # Toto 专属
```

### 模型权重
```
/home/share/llm_models/
  AutonLab/MOMENT-1-large/    # 下载后放置
  Datadog/Toto-Open-Base-1.0/ # 下载后放置
```

### 适配器设计
- 每个新算法一个独立适配器脚本
- 通过 subprocess + PYTHONNOUSERSITE=1 调用
- 标准输入输出格式（与现有 Timer/ChatTS 一致）

---

## 四、关键洞察

1. **Timer 的优势在于"时序预测→残差检测"范式**，MOMENT 和 Toto 采用类似范式但规模更大
2. **ChatTS 的优势在于"LLM 理解时序语义"**，但受限于 JSON 解析的稳定性
3. **TSB-AD 的发现**：简单方法有时不输于复杂模型——这与我们的 IForest 高评分一致
4. **零样本能力**是工业场景的关键需求（436 个点位不可能逐个微调）

Sources:
- [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885)
- [MOMENT HuggingFace](https://huggingface.co/AutonLab/MOMENT-1-large)
- [Toto by Datadog](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/)
- [Toto HuggingFace](https://huggingface.co/Datadog/Toto-Open-Base-1.0)
- [TSB-AD Benchmark](https://github.com/TheDatumOrg/TSB-AD)
- [Awesome Anomaly Detection Foundation Models](https://github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models)

# tsad-research

工业传感器时序异常检测算法研究与 Benchmark 建设。

在 436 个真实工业传感器点位上，系统评测了 **6 种检测范式、25+ 种方法/配置**，建立了完整的自动化评测流水线和双维度评估体系。

## 核心结论

**Timer+Wavelet 融合**达到最高评分 **0.767**（新 SOTA），通过率 97%。

| # | 方法 | 评分 | 范式 | 速度 |
|---|------|------|------|------|
| 1 | **Timer+Wavelet 融合** | **0.767** | 预测+信号处理 | ~60s/点 |
| 2 | Timer-84M | 0.764 | 预测→残差 | ~60s/点 |
| 3 | SENSE 选择性集成 | 0.764 | 智能路由 | - |
| 4 | Timer+FreqPatch | 0.763 | 预测+频率域 | ~60s/点 |
| 5 | IForest | 0.670 | 统计 | 1.5s/点 |
| 6 | ChatTS-14B | 0.542 | LLM | 11s/点 |
| 7 | MOMENT-1-large | 0.500 | TSFM 重建 | 8.5s/点 |
| 8 | Qwen3-VL-8B | 0.447 | VLM 目标检测 | 14s/点 |

**关键发现**：
- Timer 的"预测→残差"范式是工业 TSAD 最优路径
- 信号处理方法（小波/FFT）与 Timer 互补融合可突破单模型上限
- 图像化方法（Qwen-VL/GAF+ConvAE）全面不如数值方法
- 简单统计方法（IForest 0.670）性价比极高

## 项目结构

```
tsad-research/
├── benchmark/                      # Benchmark 框架（独立自治）
│   ├── config.yaml                 # 算法注册表
│   ├── runner.py                   # 子进程驱动调度器
│   ├── evaluator.py                # 双维度评估器
│   ├── auto_scorer.py              # 自动评分器
│   ├── compare_report.py           # 横向对比报告
│   ├── sensor_type_strategy.py     # 传感器类型分层策略
│   ├── param_sweep.py              # 参数网格搜索
│   ├── lib/                        # 核心检测库（从 ts-iteration-loop 抽取，独立运行）
│   │   ├── timer_detect.py         # Timer/Sundial 检测
│   │   ├── chatts_detect.py        # ChatTS 检测
│   │   └── signal_utils.py         # 降采样工具
│   └── adapters/                   # 16 个算法适配器
│       ├── timer_adapter.py        # Timer/Sundial
│       ├── chatts_adapter.py       # ChatTS（含 LoRA）
│       ├── moment_adapter.py       # MOMENT-1-large
│       ├── statistical_adapter.py  # IForest/3-Sigma/MAD/IQR
│       ├── wavelet_multiscale_adapter.py  # 小波多尺度
│       ├── freq_patch_adapter.py   # 频率域 Patching
│       ├── gaf_convae_adapter.py   # GAF+ConvAutoencoder
│       ├── opencv_edge_adapter.py  # OpenCV 边缘检测
│       ├── ensemble_detector.py    # 集成检测器
│       ├── sense_ensemble.py       # SENSE 选择性集成
│       ├── teacher_student.py      # Teacher-Student 伪标签
│       └── ts2vec_adapter.py       # TS2Vec 对比学习
├── qwen/                           # Qwen-VL 目标检测实验
│   ├── build_grounding_dataset.py  # bbox 数据集构建（337 训练 + 99 测试）
│   ├── inference_grounding.py      # VL 推理
│   ├── finetune_grounding.py       # LoRA 微调
│   └── evaluate_grounding.py       # Grounding 评估（IoU/P/R/F1）
├── data/                           # 数据（gitignore 排除大文件）
│   ├── adtk_hbos_old/              # 436 点位 ADTK 推理结果（46GB）
│   └── cleaned/                    # 评分数据 + 训练/测试集划分
├── results/                        # 评测结果
│   ├── predictions/                # 各算法推理结果（16 个子目录）
│   ├── auto_scores.json            # 全算法排名
│   └── *.json                      # 各类分析报告
├── docs/                           # 文档
│   ├── experiment_report.md        # 完整实验报告（537 行）
│   ├── paper_survey.md             # 论文调研（14 篇，6 方向）
│   ├── challenges_and_research_directions.md  # 难点分析与研究方向
│   ├── progress_log.md             # 研发进展日志
│   └── ...
└── scripts/                        # 数据处理脚本
```

## 评估体系

由于 436 个点位无全局 Ground Truth，采用双维度评估：

- **维度 A（主观拟合）**：基于 100 个评分点位（5 位专家 × 0/0.5/1 打分），自动评分器预测人工评分（Timer 偏差 0.092）
- **维度 B（物理约束）**：异常率硬限（0.1%-15%）、时序粘性、跳变率
- **维度 C（Grounding）**：Qwen-VL 专用，Precision/Recall/F1/IoU

## 数据划分

| 集合 | 用途 | 点位数 |
|------|------|--------|
| 测试集 | 仅评估，**禁止训练** | 99（有人工评分） |
| 训练集 | 微调/适配 | 337 |

## 覆盖的检测范式

| 范式 | 代表方法 | 最佳评分 |
|------|---------|---------|
| 预测→残差 | Timer-84M | 0.764 |
| 预测+信号处理融合 | Timer+Wavelet | **0.767** |
| 统计 | IForest | 0.670 |
| LLM 文本 | ChatTS-14B | 0.542 |
| TSFM 重建 | MOMENT | 0.500 |
| VLM 目标检测 | Qwen3-VL-8B | 0.447 |
| 计算机视觉 | OpenCV Edge | 0.488 |
| GAF+CNN | ConvAutoencoder | 0.500 |
| 对比学习 | TS2Vec | 0.450 |
| 自监督注意力 | TransformerKL | - |

## 环境要求

- 各算法在独立 conda 环境中运行（timer, chatts, qwen_tune, moment 等）
- 通过 subprocess 调用，`PYTHONNOUSERSITE=1` 隔离
- 模型权重从 `/home/share/llm_models/` 加载
- GPU: NVIDIA RTX 6000 Ada (48GB) × 2

## 文档

| 文件 | 内容 |
|------|------|
| `docs/experiment_report.md` | 完整实验报告：25+ 方法逐一分析 |
| `docs/paper_survey.md` | 14 篇论文调研，CATCH/TSB-AutoAD/AXIS 等 |
| `docs/challenges_and_research_directions.md` | 难点分析 + 6 个研究方向 |
| `docs/progress_log.md` | 研发日志 |
| `docs/research_findings.md` | 研究发现与技术建议 |

*注：受脱敏规定限制，不包含原始传感器 CSV/Excel 数据。*

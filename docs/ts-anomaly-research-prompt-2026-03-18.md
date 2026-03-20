# 工业传感器时序异常检测研究与 Benchmark 建设

**实验工程目录**：`/home/douff/my_project/tsad-research`（独立自治，不依赖外部项目）

---

## 一、双轨并行的核心目标

1. **现有算法复现与调优**：梳理盘点 Timer, ChatTS, Qwen, ADTK 等算法，在统一框架下复现并摸高其检测上限。
2. **新算法调研与新体系研发**：跳出现有体系藩篱，调研并引入时序大模型（TSFM）、自监督掩码、目标检测等全新范式方案。
3. **底座 Benchmark 建设**：搭建统一数据流接口、非监督评判指标体系的竞技验证场。

## 二、评价体系

- **维度A-主观经验拟合（核心 100 评分点）**：利用自动评分器拟合工业专家评分。已知局限：对 ChatTS 偏差较大（+0.206）。
- **维度B-物理常识约束（泛化 436 全量点）**：异常率硬限（0.1%-15%）、时序粘性、跳变率。
- **维度C-Grounding 评估（VL 目标检测专用）**：Precision/Recall/F1/IoU。
- **异常类型分层评估**（待建设）：点异常、阶跃、漂移、方差突变分别统计。

## 三、研发规则（红线）

1. **环境沙盒绝对隔离**：各算法独立 conda 环境，新算法新建环境。
2. **算法接口松耦合**：通过 `subprocess` 子进程调用，文件 I/O 通信。
3. **大文件剥离**：模型权重从 `/home/share/llm_models` 加载。
4. **代码自治**：核心检测模块已抽取到 `benchmark/lib/`，**不再依赖外部项目路径**。
5. **数据划分严格**：100 评分点位 = 测试集（禁止训练），337 剩余点位 = 训练集。
6. **审计提交**：关键节点 git commit 留档。

## 四、项目架构

```
tsad-research/                      # 独立自治的研究项目
├── benchmark/
│   ├── config.yaml                 # 算法注册表
│   ├── runner.py                   # 主调度器
│   ├── evaluator.py                # 双维度评估器
│   ├── auto_scorer.py              # 自动评分器
│   ├── compare_report.py           # 横向对比
│   ├── sensor_type_strategy.py     # 分层检测策略
│   ├── param_sweep.py              # 参数扫描
│   ├── lib/                        # 核心检测库（已从 ts-iteration-loop 抽取）
│   │   ├── timer_detect.py         # Timer/Sundial 检测
│   │   ├── chatts_detect.py        # ChatTS 检测
│   │   ├── signal_utils.py         # 降采样等工具
│   │   └── patch_transformers.py   # transformers 兼容补丁
│   └── adapters/                   # 14 个算法适配器
├── qwen/                           # Qwen-VL 目标检测实验
│   ├── build_grounding_dataset.py
│   ├── inference_grounding.py
│   ├── finetune_grounding.py
│   └── evaluate_grounding.py
├── data/                           # 数据（.gitignore 排除大文件）
├── results/                        # 评测结果
└── docs/                           # 文档
```

## 五、里程碑进度

| 里程碑 | 状态 | 交付 |
|--------|------|------|
| M1 现状盘点 | **完成** | Benchmark 框架 + ADTK baseline |
| M2 量化指标 | **完成** | 双维度评估 + 自动评分 + 参数扫描 |
| M3 新范式引入 | **完成** | 6 范式 20+ 方法评测，Timer 确认 SOTA(0.764) |
| M4 全量泛化 | 进行中 | 测试集 99 点完成，训练集 337 点待 Timer 推理 |
| M5 业务收敛 | 待开始 | 最佳方案封装 + 技术白皮书 |

## 六、当前 SOTA 与核心结论

**Timer-84M（预测→残差范式）= 绝对 SOTA**，评分 0.764，领先 IForest（0.670）14%。

**已证伪的方向**：
- 图像化方法（Qwen-VL/GAF+ConvAE/OpenCV）全面不如数值方法
- MOMENT（通用 TSFM）异常检测能力不足（异常率仅 0.03%）
- Timer/Sundial 在标准流程下输出完全一致

## 七、下一阶段研发方向

### 数据侧
- Timer 伪标签替代 ADTK（提升训练标注质量）
- 异常类型标注（点异常/阶跃/漂移/方差突变）
- 更多评审人标注扩大测试集

### 算法侧
- 全量 436 点位 Timer 推理 + 分层策略验证
- ChatTS LoRA 微调（改善漏检）
- 对比学习（TS2Vec）/ 自监督掩码（PatchTST）新范式
- 多变量关联检测（GNN）
- Timer+ChatTS 可解释性融合（精确检测+语义解释）

### 评估侧
- 按异常类型分层评估
- 检测延迟指标
- 误报/漏报非对称成本建模

# 研发进展日志

> 项目：工业传感器时序异常检测研究与 Benchmark 建设
> 项目路径：`/home/douff/my_project/tsad-research`

---

## 2026-03-19 研发进展

### 一、数据集划分规则（重要）

| 用途 | 数据集 | 点位数 | 说明 |
|------|--------|--------|------|
| **测试集** | 100 个有人工评分的点位 | 100 | `data/cleaned/evaluated_points.txt`，仅用于评估，**禁止用于训练** |
| **训练集** | 全量 436 中排除测试集的点位 | 336 | `data/adtk_hbos_old/` 中非测试集点位，可用于微调/适配 |

### 二、里程碑完成进度

#### M1：现状盘点立根基 — 完成

- 构建了 `benchmark/` 子进程驱动的评测流水线
- 算法注册表 `config.yaml`，支持松耦合的适配器架构
- ADTK+HBOS 436 点位 baseline 评估完成
- 环境隔离方案：`PYTHONNOUSERSITE=1` + 独立 conda

#### M2：量化指标逻辑上马 — 完成

- 双维度评估器：维度A（100评分点主观拟合）+ 维度B（436全量物理约束）
- 自动评分器：基于规则预测人工评分，Timer 偏差 0.092
- 特征-评分关联分析、评分预测报告
- 参数扫描实验（threshold_k 5 种配置）

#### M3：新范式沙盒引入 — 进行中

**已完成推理的算法：**

| 算法 | 类型 | 环境 | 预测均分 | 通过率 | 良好率 | 状态 |
|------|------|------|---------|--------|--------|------|
| Timer-84M | TSFM（预测→残差） | timer conda | 0.764 | 94.9% | 87.9% | 99/99 完成 |
| ChatTS-14B | LLM（文本理解时序） | chatts conda | 0.542 | 73.7% | 0.0% | 100/100 完成 |
| MOMENT-1-large | TSFM（重建→残差） | moment conda | 0.500 | 100% | 0.0% | ~56/99 运行中 |
| IForest | 统计 | base | 0.670 | 77.8% | 50.5% | 99/99 完成 |
| 3-Sigma | 统计 | base | 0.436 | 40.4% | 21.2% | 99/99 完成 |
| MAD | 统计 | base | 0.498 | 47.5% | 30.3% | 99/99 完成 |
| IQR | 统计 | base | 0.529 | 55.6% | 36.4% | 99/99 完成 |
| Weighted Ensemble | 集成 | base | 0.717 | 85.2% | 74.1% | 27 点完成 |
| Qwen-VL | VLM（看图识异常） | qwen_tune conda | - | - | - | 适配器就绪 |

**已有人工评分对照：**

| 算法 | 实际人工均分 | 自动预测均分 | 偏差 |
|------|------------|------------|------|
| Timer | 0.672 | 0.764 | +0.092 |
| ChatTS | 0.336 | 0.542 | +0.206 |
| Qwen-VL | 0.225 | 待测 | - |

### 三、新算法调研结论

#### MOMENT 和 Toto 的定位澄清

**MOMENT**（CMU, ICML'24）：
- **主任务**：时序预测、分类、异常检测、补全（多任务通用模型）
- **异常检测方式**：重建范式——重建原始时序，残差大 = 异常
- **问题**：论文自身的 benchmark 显示 **K-NN 在异常检测上击败了所有深度模型**（包括 MOMENT）
- **结论**：MOMENT 不是异常检测专用模型，其 AD 能力是预训练副产品，不是主要竞争力

**Toto**（Datadog, 2025）：
- **主任务**：可观测性场景的时序预测
- **异常检测方式**：基于预测区间（prediction intervals）的动态阈值
- **本质**：与 Timer 一样是"预测→残差→阈值"范式，但面向 IT 运维而非工业传感器
- **结论**：可以尝试，但不太可能在工业场景上超越 Timer

#### 真正的 TSAD 专用方向

时序异常检测的主流方法分 4 类：
1. **预测型**（Timer, Sundial, Toto）：预测未来值，偏差 = 异常
2. **重建型**（MOMENT, Autoencoder）：重建输入，重建误差 = 异常
3. **表示学习型**（对比学习、自监督掩码）：学习正常模式的表示
4. **混合型**：结合多种范式

**Timer 已经是预测型 SOTA**。要真正突破，应关注：
- **表示学习型**：如 TS2Vec、TNC 等对比学习方法
- **自监督掩码型**：如 PatchTST 的 masked modeling
- **工业专用改进**：针对传感器类型的分层检测策略

### 四、关键发现

1. **Timer 检测更精细**：平均 599 个短簇（139 点/簇） vs ADTK 21 个长簇（19356 点/簇）
2. **ChatTS 异常率偏低**：平均 1.1%（Timer 3.3%），22/100 零异常——漏检严重
3. **特征-评分相关性弱**（max r=0.24）：人工评分依赖视觉合理性而非统计指标
4. **Timer/Sundial 在标准流程下输出完全一致**：降采样+MAD阈值抹平了模型差异
5. **简单统计方法（IForest）预测评分 0.670**：与 Timer 差距不大，印证 TSB-AD 发现

### 五、项目文件结构

```
tsad-research/
├── benchmark/                    # Benchmark 框架
│   ├── config.yaml              # 算法注册表
│   ├── runner.py                # 主调度器
│   ├── evaluator.py             # 双维度评估器
│   ├── auto_scorer.py           # 自动评分器
│   ├── score_predictor.py       # 评分预测分析
│   ├── compare_report.py        # 对比报告生成
│   ├── timer_score_analysis.py  # Timer 深层分析
│   ├── param_sweep.py           # 参数扫描实验
│   ├── batch_inference.py       # 批量推理
│   ├── run_statistical_baselines.py
│   └── adapters/
│       ├── timer_adapter.py     # Timer/Sundial 适配器
│       ├── timer_batch_adapter.py
│       ├── chatts_adapter.py    # ChatTS 适配器
│       ├── chatts_batch_adapter.py
│       ├── qwen_vl_adapter.py   # Qwen-VL 适配器
│       ├── moment_adapter.py    # MOMENT 适配器（新范式）
│       ├── statistical_adapter.py # IForest/3Sigma/MAD/IQR
│       ├── ensemble_detector.py # 集成检测器
│       └── adtk_hbos_adapter.py
├── data/
│   ├── adtk_hbos_old/           # 436 个 ADTK 推理 CSV（46GB）
│   ├── cleaned/                 # 清洗后评分数据
│   └── features/                # 提取的特征数据
├── results/
│   ├── predictions/             # 各算法推理结果
│   │   ├── timer/              # 99 个点位
│   │   ├── chatts/             # 100 个点位
│   │   ├── moment/             # ~56 个点位（运行中）
│   │   ├── iforest/3sigma/mad/iqr/  # 各 99 个点位
│   │   └── ensemble_*/         # 集成结果
│   ├── benchmark_summary.json
│   ├── comparison_report.json
│   ├── auto_scores.json
│   └── score_prediction_report.json
├── docs/
│   ├── ts-anomaly-research-prompt-2026-03-18.md  # 研发需求文档
│   ├── data_preparation_summary.md               # 数据准备总结
│   ├── research_findings.md                      # 研究发现与建议
│   ├── new_algorithm_research.md                 # 新算法调研报告
│   └── progress_log.md                           # 本文件
└── scripts/                     # 数据处理脚本（已有）
```

### 六、Git 提交记录

```
ffa5231 feat: integrate ChatTS, Qwen-VL, and MOMENT into benchmark pipeline
e58b0c1 experiment: Timer threshold_k parameter sweep
cc3489f analysis: Timer inference features vs human scores deep correlation
b233ab1 feat: Timer-84M full evaluation on 99 scored points
bae3d71 feat: add ensemble detector and auto-scorer
25b468c feat: add statistical method baselines
f91e4bb feat: build benchmark framework with subprocess-driven pipeline
c69ac04 docs: backup the original prompt document before refactor
8e6d77a docs: huge refactor and refine for the research prompt
faf691e docs: add shared llm models path and environment constraints
9bff567 first commit: add research plan, data prep scripts and documentation
```

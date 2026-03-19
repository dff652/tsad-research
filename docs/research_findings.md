# 研究发现与建议

> 更新时间：2026-03-19
> 项目路径：`/home/douff/my_project/tsad-research`

---

## 一、环境兼容性问题与解决方案

### 1.1 transformers 版本冲突

**问题**：用户级 `~/.local/lib/python3.12/site-packages/transformers` (v4.52) 会覆盖 conda 环境内的版本（如 timer 环境的 v4.40），导致 Timer/Sundial 模型推理时 `DynamicCache.get_max_length()` 报错。

**解决方案**：在 subprocess 调用时设置 `PYTHONNOUSERSITE=1`，禁止加载用户级 site-packages。

**建议**：长期方案应清理 `~/.local/lib/python3.12/site-packages/` 中的冗余包，或为每个 conda 环境显式设置 `PYTHONNOUSERSITE=1`。

### 1.2 Timer/Sundial 同架构族

Timer-84M 和 Sundial-128M 使用完全相同的推理接口（`AutoModelForCausalLM` + `generate()`），差异仅在模型权重。可共用同一适配器脚本和 conda 环境，通过 `--model-path` 参数区分。

---

## 二、数据层面发现

### 2.1 数据规模

- 全量 CSV 数据约 **46GB**（436 个点位），不适合全量复制
- 已提取的特征文件 `all_points_features.csv` 包含所有必要的聚合统计
- 评估应优先使用特征文件，仅在需要精细分析时读取原始 CSV

### 2.2 异常率分布极不均匀

| 类型 | 平均异常率 | 特点 |
|------|-----------|------|
| Current/Power | 27.5% | 极高，可能存在传感器本身问题 |
| AC_Controller | 9.5% | 偏高 |
| Temperature | 6.6% | 中等 |
| SOV/Discrete | 0.2% | 极低 |
| Speed | 0.0% | 无异常 |

**建议**：评估指标应按传感器类型分层，不宜用统一阈值。

### 2.3 82 个零异常点位

全量 436 个点位中，82 个完全没有异常标记。需要确认这些是否为真正的正常数据，还是 ADTK+HBOS 的漏检。

---

## 三、评估体系发现

### 3.1 特征-评分相关性弱

所有特征与人工评分的相关系数 |r| < 0.22，说明人工评分不是简单的统计规则可以替代的。评分更依赖于：
- 异常检测结果的"视觉合理性"（是否与图形趋势吻合）
- 领域知识（某些传感器类型的特有模式）
- 主观偏好差异（评审人间一致性仅 0.18-0.35）

### 3.2 Timer 的"高异常率偏好"

Timer 在异常率 > 20% 的点位上评分最高 (0.793)，而在异常率 < 0.1% 的点位上评分较低 (0.665)。这暗示 Timer 擅长处理复杂多变的工业时序，而在"平静"数据上可能产生误报。

### 3.3 ADTK+HBOS 物理特性

- 跳变率极低 (0.013)：异常标记非常"粘"，倾向于标记大段连续区间
- 平均异常簇长度 23840 点：可能过度标记连续区间
- 物理约束检查通过（全量异常率 4.68%）

---

## 四、架构建议

### 4.1 Benchmark 流水线已验证的设计

```
config.yaml           → 算法注册表
adapters/              → 各算法独立适配器（subprocess 调用）
evaluator.py           → 双维度评估器（维度A + 维度B）
runner.py              → 主调度器
batch_inference.py     → 批量推理脚本
score_predictor.py     → 量化评分预测器
```

### 4.2 后续扩展方向

1. **新算法接入**：只需添加 adapter 脚本 + config.yaml 注册，无需修改框架
2. **多 GPU 并行**：batch_inference 可扩展为多 GPU 并行推理
3. **增量评估**：新增评估维度只需在 evaluator.py 中添加方法
4. **可视化对比**：需增加异常检测结果的可视化对比工具

### 4.3 推理性能参考

- Timer/Sundial 单点位推理：~60 秒（降采样到 10000 点，lookback=256）
- 100 个评分点位全量推理：~100 分钟
- 436 个全量点位推理：~7 小时

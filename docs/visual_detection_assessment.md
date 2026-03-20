# 时序异常检测的视觉/目标检测方向评估

> 更新：2026-03-20

---

## 一、Qwen-VL 目标检测效果总结

### 实测数据

| 指标 | Qwen3-VL-8B 零样本 | 微调后 | Timer (SOTA) |
|------|-------------------|--------|-------------|
| 自动评分 | 0.446 | 0.447 | **0.764** |
| 异常率 | 31.4%（过检） | 30.5% | 3.3% |
| Grounding F1 | 0.167 | 0.173 | - |
| Grounding IoU | 0.315 | 0.329 | - |
| TP 匹配数 | 74/968 | 86/968 | - |

### 核心问题

1. **VLM 缺乏时序领域知识**：把正常周期波动标为异常，22/99 点位输出全图大框
2. **像素精度不足**：4000px 图像中 1px ≈ 数百个数据点，精确定位异常边界困难
3. **训练标注质量差**：用 ADTK（评分 0.586）标注训练，上限受限
4. **微调效果有限**：3 epoch + 223 样本带来 +3.6% F1 提升

### 结论

Qwen-VL 的"看图找异常"在当前实现下**不如数值方法**（Timer/统计）。VLM 的优势在"理解语义"而非"精确定位像素"，与异常检测的精度要求不匹配。

---

## 二、其他目标检测模型方案对比

### 方案 A：YOLO/DETR 直接检测时序图（当前 Qwen-VL 方案的轻量替代）

| 项 | 评估 |
|----|------|
| 模型 | YOLOv8/v11, RT-DETR |
| 输入 | 时序曲线图（同 Qwen-VL） |
| 标注 | COCO 格式 bbox |
| 优势 | 推理极快（<10ms/图），可实时部署 |
| 劣势 | 与 Qwen-VL 面临同样的像素精度问题 |
| 预期效果 | 略好于 Qwen-VL（专用检测器 vs 通用 VLM），但仍受限于图像化精度 |
| 接入难度 | 低（ultralytics 一键训练） |

### 方案 B：GAF (Gramian Angular Field) + CNN/Autoencoder（推荐）

| 项 | 评估 |
|----|------|
| 原理 | 将时序转为 GAF 图像（保留完整数值信息），用 ConvAutoencoder 重建，重建误差 = 异常 |
| 输入 | 固定长度窗口的 GAF 图（如 256×256） |
| 优势 | **不损失数值精度**（像素值 = 时间步对之间的角度关系），已有论文报告 F1≈99% |
| 劣势 | 窗口化处理，需要滑动窗口；GAF 计算有开销 |
| 预期效果 | **有望显著优于 Qwen-VL**，可能接近 Timer |
| 接入难度 | 中等（pyts 已安装，需训练 ConvAE） |

### 方案 C：OpenCV 边缘检测增强（已有 cvd.py 基础）

| 项 | 评估 |
|----|------|
| 当前效果 | 评分 0.488，通过率 91.9%，仅检测跳变 |
| 改进方向 | 结合多尺度 Hough 变换 + 自适应阈值 |
| 优势 | 极快（0.5s/点），无需 GPU |
| 劣势 | 只能检测"突变/跳变"，无法检测漂移/方差变化 |

### 方案 D：Transformer+KL 自监督（已有 anml_trsf.py 基础）

| 项 | 评估 |
|----|------|
| 当前效果 | 单点测试 rate=0.11%，训练 609s |
| 改进方向 | 用训练集 337 点位批量预训练通用模型（而非每点独立训练） |
| 优势 | 自监督新范式，不依赖标注 |
| 劣势 | 训练慢（每点 10 分钟） |

---

## 三、推荐优先级

| 优先级 | 方案 | 理由 |
|--------|------|------|
| **P0** | **GAF + ConvAutoencoder** | 保留数值精度，文献报告效果好，pyts 已就绪 |
| P1 | YOLO 快速检测器 | 推理极快，可作为实时部署方案 |
| P1 | Transformer+KL 预训练优化 | 改为全局预训练（而非逐点），可大幅加速 |
| P2 | OpenCV 多尺度增强 | 补充跳变检测能力 |
| 降级 | Qwen-VL 进一步微调 | 投入产出比低，除非有 Timer 级别标注 |

---

## 四、各方案投入产出比

```
        效果预期
          ↑
   GAF+  |  ★ GAF+ConvAE
  Conv   |
         |        ★ TransfKL(预训练优化)
         |   ★ YOLO
         |
   Qwen  |  ★ Qwen微调(Timer标注)
  微调   |
         |  ★ OpenCV增强
         |  ★ Qwen微调(ADTK标注) ← 当前
         +--------------------------------→ 投入时间
             少                        多
```

Sources:
- [GAF + Anomaly Detection (99% F1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679659/)
- [pyts GAF implementation](https://pyts.readthedocs.io/en/stable/generated/pyts.image.GramianAngularField.html)
- [YOLO-World Open-Vocabulary Detection](https://github.com/AILab-CVC/YOLO-World)
- [Ultralytics YOLO Anomaly Detection](https://www.ultralytics.com/blog/vision-ai-for-anomaly-detection-a-quick-overview)

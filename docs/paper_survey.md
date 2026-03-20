# 时序异常检测论文调研

> 按推荐度排序，聚焦可落地到本项目的方向

---

## 方向 1：无标签/弱标签异常检测（★★★）

### 1.1 TSB-AutoAD (VLDB 2025) —— 自动化模型选择

- **论文**：TSB-AutoAD: Towards Automated Solutions for Time-Series Anomaly Detection
- **来源**：[VLDB 2025](https://dl.acm.org/doi/10.14778/3749646.3749699)
- **代码**：[github.com/TheDatumOrg/TSB-AutoAD](https://github.com/TheDatumOrg/TSB-AutoAD)

**核心思想**：系统评估 20 种独立方法 + 70 种变体的自动化异常检测方案，发现超过一半的自动方案不如随机选择，75% 不如全局最优单模型。提出 SENSE（Selective Ensembling），结合模型选择和集成优势。

**与本项目关联**：
- 我们的实验也发现集成方法（Weighted Ensemble 0.717）不显著优于单模型（Timer 0.764）
- SENSE 的"选择性集成"思路可直接应用——按传感器类型路由到最优单模型
- 其评估框架（跨数据集、跨异常类型）可参考

**可落地方案**：实现 SENSE 框架，用传感器类型作为路由条件

### 1.2 Teacher-Student 伪标签迭代

- **相关工作**：Asymmetric Student-Teacher Networks (WACV 2023), STAD-GAN
- **来源**：[WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.pdf)

**核心思想**：Teacher 模型（如 Timer）生成伪标签 → Student 模型在伪标签上训练 → Student 可能发现 Teacher 漏检的异常 → 迭代提升

**与本项目关联**：
- Timer（0.764）可作为 Teacher
- ChatTS/IForest 可作为 Student
- 337 个训练集点位可用 Timer 伪标签训练

**可落地方案**：
1. 在 337 训练点位上跑 Timer → 生成伪标签
2. 用伪标签训练轻量 Student（如改进的 IForest/MAD）
3. Student 在测试集上推理，对比 Timer

### 1.3 VLT-Anomaly (2025) —— VAE+Transformer 增强

- **论文**：Unsupervised Anomaly Detection in Time Series Data via Enhanced VAE-Transformer
- **来源**：[sciopen.com](https://www.sciopen.com/article/10.32604/cmc.2025.063151)

**核心思想**：重新设计 VAE 网络结构使其更适合异常检测的数据重建，结合 Transformer 捕获长程依赖。

**与本项目关联**：我们的 GAF+ConvAE 重建方法失败（区分度不足），VAE 的概率建模可能提供更好的异常/正常区分

---

## 方向 2：多尺度检测（★★★）

### 2.1 小波分解 + 图注意力网络 (2024)

- **论文**：Anomaly Detection for Multivariate Time Series in IoT Using Discrete Wavelet Decomposition and Dual Graph Attention Networks (Meta-MWDG)
- **来源**：[ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0167404824003808)

**核心思想**：离散小波分解将时序分为多个频率分量 → 双图注意力网络分别建模特征相关性和时间依赖性 → 多尺度融合检测

**与本项目关联**：
- Timer 使用固定 256 窗口，无法同时捕获快速尖峰和慢速漂移
- 小波分解可将信号分为高频（尖峰）和低频（趋势）分别处理
- STL 分解已在业务平台中使用（signal_processing.py），可扩展为小波版

**可落地方案**：
1. 对降采样数据做 DWT 分解（高频 + 低频）
2. 高频分量用 3-Sigma/MAD 检测尖峰
3. 低频分量用 Timer 检测趋势漂移
4. 融合两路结果

### 2.2 WaveLST-Trans (2025) —— 小波+LSTM+Transformer

- **论文**：Anomaly Detection and Risk Early Warning System Based on WaveLST-Trans
- **来源**：[icck.org](https://www.icck.org/article/abs/tetai.2025.191759)

**核心思想**：小波变换提取多尺度特征 → LSTM 捕获短期依赖 → Transformer 捕获长期依赖 → 特征融合层整合

---

## 方向 3：频率域 Patching（★★★ 最新 ICLR 2025）

### 3.1 CATCH (ICLR 2025) —— 频率 Patching + 通道融合

- **论文**：CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching
- **来源**：[ICLR 2025](https://openreview.net/forum?id=m08aK3xxdJ)
- **代码**：[github.com/decisionintelligence/CATCH](https://github.com/decisionintelligence/CATCH)

**核心思想**：
1. 将频率域分割为频率 band（Frequency Patching），增强对细粒度频率特征的捕捉
2. Channel Fusion Module (CFM)：patch-wise mask 生成器 + masked-attention 机制
3. 双层多目标优化驱动 CFM 发现合适的 patch-wise 通道关联

**与本项目关联**：
- 直接解决我们发现的 GAF 编码失败问题——频率域 patching 比时域 GAF 更好地保留异常信号
- 支持多变量（通道关联），可扩展到多传感器场景
- ICLR 2025 接收，学术认可度高

**可落地方案**：
1. 克隆 CATCH 代码，新建 conda 环境
2. 用 337 训练点位的数据训练
3. 在 99 测试点位评估
4. 对比 Timer（单变量预测型 vs 多变量频率型）

**优先级：★★★ 最高——ICLR 2025 顶会，开源代码，直接解决我们的技术瓶颈**

### 3.2 TimeVQVAE-AD (2024) —— 时频域 VQ-VAE

- **论文**：TimeVQVAE for Time Series Anomaly Detection
- **代码**：[github.com/ML4ITS/TimeVQVAE-AnomalyDetection](https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection)

**核心思想**：用 VQ-VAE（向量量化变分自编码器）在时频域进行生成式建模，通过重建质量判断异常

---

## 方向 4：可解释异常检测（★★☆）

### 4.1 AXIS (2025) —— LLM 可解释异常检测

- **论文**：AXIS: Explainable Time Series Anomaly Detection with Large Language Models
- **来源**：[arxiv 2025](https://arxiv.org/abs/2509.24378)

**核心思想**：冻结的 LLM 配合三种协同输入：
1. Symbolic numeric hint（数值锚定）
2. Context-integrated step-aligned hint（细粒度动态）
3. Task-prior hint（全局任务先验）
使 LLM 成为零样本异常检测器，同时输出可解释的异常描述。

**与本项目关联**：
- 直接对标 ChatTS 的方向——但 AXIS 更系统化
- 可与 Timer 结合：Timer 检测 + AXIS/ChatTS 解释

### 4.2 Can LLMs Understand Time Series Anomalies? (ICLR 2025)

- **来源**：[ICLR 2025](https://openreview.net/forum?id=LGafQ1g2D2)

**关键发现**：
- **LLMs 理解时序图像优于理解时序文本**——这与我们 Qwen-VL 的实验一致
- LLMs 的显式推理 prompt 并不能增强检测性能
- 适当的 prompt 和 scaling 策略是关键

**与本项目关联**：验证了我们的发现——ChatTS（文本范式）效果不如视觉范式的理论预期，但实际实现中图像精度损失抵消了这一优势

### 4.3 Multimodal LLMs for TSAD (2025)

- **来源**：[arxiv](https://arxiv.org/pdf/2502.17812)

**核心发现**：多模态 LLM 在时序异常检测中的潜力尚未充分发掘，需要更好的时序→视觉编码策略

---

## 方向 5：多变量关联检测（★★☆）

### 5.1 Directed Hypergraph Neural Networks (2025)

- **论文**：Multivariate Time Series Anomaly Detection Using Directed Hypergraph Neural Networks
- **来源**：[Taylor & Francis 2025](https://www.tandfonline.com/doi/full/10.1080/08839514.2025.2538519)

**核心思想**：用有向超图建模传感器间的多对多关系（不仅是成对关系），检测违反这些关系的异常

### 5.2 DVGCRN (2025 Survey 推荐)

- **核心思想**：Embedding-Guided Probabilistic Generative Network + 自适应变分图卷积循环网络，同时建模空间和时间关联

### 5.3 iADCPS (2025) —— 增量元学习

- **论文**：iADCPS: Time Series Anomaly Detection for Evolving Cyber-physical Systems via Incremental Meta-learning
- **来源**：[arxiv 2025](https://arxiv.org/abs/2504.04374)

**核心思想**：增量元学习应对概念漂移（传感器行为随时间变化），在 PUMP/SWaT/WADI 数据集上 F1 分别达到 99.0%/93.1%/78.7%

**与本项目关联**：工业传感器可能存在概念漂移（季节变化、设备老化），增量学习可应对

---

## 方向 6：对比学习/自监督（★★☆）

### 6.1 TS2Vec (AAAI 2022, 持续影响)

- **论文**：TS2Vec: Towards Universal Representation of Time Series
- **来源**：[AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20881)
- **代码**：[github.com/yuezhihan/ts2vec](https://github.com/yuezhihan/ts2vec)

**异常检测方式**：将时序片段的表示计算两次（有/无最后一个观测值的 mask），两次表示的距离 = 异常分数

**2024 改进**：SoftCLT（Soft Contrastive Learning, ICLR 2024）在 TS2Vec 基础上改进，F1 提升约 2%

### 6.2 PatchAD (2024) —— Patch + 对比学习

- **核心思想**：基于 PatchTST 概念的轻量架构，用对比学习和 Patch-based MLP-Mixer 提取时序语义特征

**与本项目关联**：PatchAD 轻量化，可能适合工业部署

---

## 综合评估：可落地性排序

| 优先级 | 方案 | 论文 | 代码 | 接入难度 | 预期效果 |
|--------|------|------|------|---------|---------|
| **P0** | **CATCH 频率 Patching** | ICLR 2025 | ✅ 开源 | 中（新环境+训练） | **高——解决 GAF 失败的根因** |
| **P0** | **SENSE 选择性集成** | VLDB 2025 | ✅ 开源 | 低（框架集成） | **中高——系统化我们的集成策略** |
| P1 | 小波多尺度 + Timer | Meta-MWDG 2024 | 可自研 | 低（scipy 小波） | 中——补充 Timer 的多尺度能力 |
| P1 | Teacher-Student 伪标签 | WACV 2023 | 可自研 | 低 | 中——提升训练标注质量 |
| P1 | AXIS/LLM 可解释 | 2025 | 部分开源 | 中 | 中——补充可解释性 |
| P2 | TS2Vec/SoftCLT 对比学习 | AAAI 2022 / ICLR 2024 | ✅ 开源 | 中 | 中——新表示范式 |
| P2 | GNN 多变量 | 2025 多篇 | ✅ 多个开源 | 高（需多变量数据） | 高但实现复杂 |

---

## 推荐研发路线

### 第一批（1-3天）：CATCH + 小波多尺度

```
1. 克隆 CATCH → 新建 conda → 训练+评估 → 对比 Timer
2. 实现小波分解+Timer 融合 → 评估多尺度提升
```

### 第二批（2-3天）：SENSE + 伪标签

```
3. 实现 SENSE 选择性集成 → 对比 Weighted Ensemble
4. Timer 伪标签 → 训练 Student → 评估迭代提升
```

### 第三批（3-5天）：可解释性 + 对比学习

```
5. Timer+ChatTS 可解释管线
6. TS2Vec/PatchAD 对比学习实验
```

---

## 参考文献

1. [TSB-AutoAD (VLDB 2025)](https://dl.acm.org/doi/10.14778/3749646.3749699)
2. [CATCH (ICLR 2025)](https://openreview.net/forum?id=m08aK3xxdJ)
3. [AXIS: Explainable TSAD with LLMs](https://arxiv.org/abs/2509.24378)
4. [Can LLMs Understand TS Anomalies? (ICLR 2025)](https://openreview.net/forum?id=LGafQ1g2D2)
5. [TS2Vec (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20881)
6. [SoftCLT (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/ccc48eade8845cbc0b44384e8c49889a-Paper-Conference.pdf)
7. [Meta-MWDG Wavelet+GAT (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0167404824003808)
8. [iADCPS Incremental Meta-learning (2025)](https://arxiv.org/abs/2504.04374)
9. [Directed Hypergraph for MTSAD (2025)](https://www.tandfonline.com/doi/full/10.1080/08839514.2025.2538519)
10. [Asymmetric Student-Teacher (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.pdf)
11. [Deep Learning TSAD Survey (ACM 2024)](https://dl.acm.org/doi/full/10.1145/3691338)
12. [TimeVQVAE-AD](https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection)
13. [PatchAD](https://arxiv.org/html/2412.05498v1)
14. [VLT-Anomaly (2025)](https://www.sciopen.com/article/10.32604/cmc.2025.063151)

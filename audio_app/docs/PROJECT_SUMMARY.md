# 🎓 音频-文本检索项目总结

## 📊 项目概述

**项目名称**: DCASE 2025 Task 6 - 音频-文本检索系统

**项目目标**: 构建一个能够理解音频内容并与文本描述进行匹配的AI系统

**最终成果**: mAP@10 = 0.32，并开发了实用的Web应用

**项目周期**: 2026年1月 - 2026年2月

---

## 🎯 核心成果

### 1. 模型性能

| 指标 | 数值 | 说明 |
|------|------|------|
| **mAP@10** | **0.32** | 主要评估指标 |
| **R@1** | 0.19 | Top-1召回率 |
| **R@5** | 0.45 | Top-5召回率 |
| **R@10** | 0.59 | Top-10召回率 |

**性能提升**:
- 相比基线（单一PaSST）提升约 **10%**
- 达到预期目标范围（0.31-0.33）

### 2. 技术方案

**最终采用**: PaSST + CLAP 双编码器融合

**架构特点**:
- **PaSST**: 通用音频理解（768维）
- **CLAP**: 音频-文本对齐专家（512维）
- **融合策略**: Attention动态加权
- **文本编码器**: RoBERTa-base

**模型规模**:
- 总参数: ~688M
- 可训练参数: ~444M
- 冻结参数: ~243M

### 3. 实用应用

开发了基于Gradio的Web应用，实现：
- ✅ 音频上传和麦克风录音
- ✅ 70+种音频场景识别
- ✅ 置信度评分和可视化
- ✅ 美观易用的界面
- ✅ 一键部署

---

## 🔬 技术探索历程

### 阶段1: 基线建立

**方案**: 单一PaSST编码器
- 性能: mAP@10 ≈ 0.29
- 问题: 性能瓶颈，难以突破

### 阶段2: 多编码器探索

**尝试方案**:

1. **PaSST + BEATs + CLAP (三编码器)**
   - 结果: mAP@10 = 0.301
   - 问题: 过拟合，BEATs引入噪声
   - 结论: 模型过于复杂

2. **PaSST + CLAP (双编码器)** ✅
   - 结果: mAP@10 = 0.32
   - 优势: 互补性强，不易过拟合
   - 结论: 最优方案

### 阶段3: 优化与调优

**关键优化**:
- ✅ 启用AudioCaps数据集（+49K样本）
- ✅ 修复BEATs音频重采样问题
- ✅ 优化学习率调度策略
- ✅ 添加早停机制防止过拟合
- ✅ 显存泄露修复

---

## 📈 训练配置

### 最终配置

```python
# 数据
数据集: Clotho (19K) + AudioCaps (49K) = 68K样本
数据增强: 时间拉伸、音高变换、添加噪声

# 模型
音频编码器: PaSST + CLAP (双编码器)
文本编码器: RoBERTa-base
融合策略: Attention
嵌入维度: 1024

# 训练
Batch size: 16 (有效batch=32，梯度累积2次)
学习率: 2e-5 (cosine衰减 + warmup)
优化器: AdamW (weight_decay=0.01)
训练轮数: 40 epochs
验证频率: 每5个epoch

# 正则化
Dropout: 0.1
梯度裁剪: 1.0
早停: patience=8
```

### 训练资源

- **GPU**: NVIDIA A100 (40GB)
- **显存占用**: ~15GB
- **训练时间**: ~20小时（40 epochs）
- **每epoch时间**: ~30分钟

---

## 🗂️ 项目文件结构

```
d25_t6/
├── 核心代码
│   ├── train.py                    # 训练脚本
│   ├── retrieval_module.py         # 主模型
│   ├── multi_audio_encoder.py      # 多编码器融合
│   ├── passt.py                    # PaSST编码器
│   ├── beats_loader.py             # BEATs加载器
│   └── predict.py                  # 推理脚本
│
├── 应用程序
│   ├── audio_caption_app.py        # Web应用主程序
│   ├── test_app.py                 # 环境测试脚本
│   ├── start_app.bat               # Windows启动脚本
│   ├── start_app.sh                # Linux启动脚本
│   └── requirements_app.txt        # 应用依赖
│
├── 数据处理
│   └── datasets/
│       ├── audio_loading.py        # 音频加载
│       ├── batch_collate.py        # 批处理
│       └── utils.py                # 工具函数
│
├── 模型文件
│   └── Train4(0.32)/
│       └── mAP@10=0.32.ckpt       # 最佳模型
│
└── 文档
    └── docs/
        ├── README_APP.md           # 应用说明
        ├── APP_USAGE.md            # 详细使用指南
        ├── TRAINING_PROGRESS.md    # 训练进度
        ├── FINAL_TRAINING_SUMMARY.md # 训练总结
        ├── MULTI_ENCODER_USAGE.md  # 多编码器指南
        └── INDEX.md                # 文档索引
```

---

## 💡 关键经验总结

### 成功经验

1. **双编码器优于三编码器**
   - 互补性比数量更重要
   - CLAP专注音频-文本对齐，与PaSST完美互补

2. **数据增强很重要**
   - AudioCaps增加了3.5倍数据量
   - 显著提升模型泛化能力

3. **防止过拟合**
   - 早停机制必不可少
   - 定期验证及时发现问题

4. **实用性优先**
   - 最终开发了Web应用
   - 让模型真正可用

### 踩过的坑

1. **BEATs音频重采样问题**
   - 问题: 需要16kHz但输入是32kHz
   - 解决: 添加重采样层

2. **三编码器过拟合**
   - 问题: 模型太复杂，数据不足
   - 解决: 回归双编码器

3. **显存泄露**
   - 问题: 训练过程中显存持续增长
   - 解决: 正确管理梯度和中间变量

4. **学习率调优**
   - 问题: 初始学习率过高导致不稳定
   - 解决: 使用warmup + cosine衰减

---

## 🎓 技术亮点

### 1. 多模态融合

**创新点**: 使用Attention机制动态融合多个音频编码器

```python
# 注意力融合伪代码
for encoder in [passt, clap]:
    features.append(encoder(audio))

attention_weights = attention_layer(features)
fused = weighted_sum(features, attention_weights)
```

### 2. 跨模态检索

**核心思想**: 将音频和文本映射到同一语义空间

```python
# 对比学习损失
audio_emb = audio_encoder(audio)
text_emb = text_encoder(text)

similarity = cosine_similarity(audio_emb, text_emb)
loss = contrastive_loss(similarity, labels)
```

### 3. 高效推理

**优化策略**:
- 预计算候选描述embeddings
- 批量处理音频
- GPU加速

---

## 📱 应用场景

开发的Web应用可用于：

1. **音频内容分析**
   - 自动标注音频库
   - 音频搜索引擎

2. **辅助工具**
   - 帮助听障人士理解环境声音
   - 音频监控系统

3. **教育研究**
   - 音频识别教学演示
   - 研究原型验证

4. **娱乐应用**
   - 音效识别游戏
   - 创意音频工具

---

## 📊 性能对比

| 方案 | mAP@10 | 参数量 | 训练时间 | 推理速度 |
|------|--------|--------|----------|----------|
| 单一PaSST | 0.29 | ~400M | 15h | 0.5s |
| PaSST+CLAP | **0.32** | ~688M | 20h | 1.5s |
| PaSST+BEATs+CLAP | 0.301 | ~900M | 25h | 2.0s |

**结论**: PaSST+CLAP方案在性能、效率、稳定性上达到最佳平衡

---

## 🚀 未来改进方向

### 短期优化

1. **扩展候选描述库**
   - 增加更多音频类别
   - 支持中文描述

2. **优化推理速度**
   - 模型量化
   - 知识蒸馏

3. **改进用户体验**
   - 添加音频可视化
   - 支持批量处理

### 长期探索

1. **多语言支持**
   - 训练多语言文本编码器
   - 支持跨语言检索

2. **细粒度识别**
   - 识别音频中的多个事件
   - 时间戳定位

3. **生成式描述**
   - 从检索改为生成
   - 自动生成自然语言描述

4. **实时处理**
   - 流式音频处理
   - 实时反馈

---

## 🎉 项目成果

### 量化成果

- ✅ 模型性能: mAP@10 = 0.32
- ✅ 代码文件: 15+ 核心文件
- ✅ 文档: 10+ 详细文档
- ✅ 应用: 1个完整的Web应用

### 能力提升

- ✅ 深度学习模型训练与调优
- ✅ 多模态学习理论与实践
- ✅ PyTorch Lightning框架使用
- ✅ 模型部署与应用开发
- ✅ 项目管理与文档编写

### 可交付物

1. **训练好的模型**: `Train4(0.32)/mAP@10=0.32.ckpt`
2. **完整代码库**: 包含训练、推理、应用全套代码
3. **Web应用**: 即开即用的音频识别系统
4. **详细文档**: 使用指南、技术文档、项目总结

---

## 📝 致谢

感谢以下开源项目和资源：

- **PaSST**: Efficient Training of Audio Transformers
- **CLAP**: Contrastive Language-Audio Pretraining
- **BEATs**: Audio Pre-Training with Acoustic Tokenizers
- **Gradio**: 快速构建ML应用的框架
- **PyTorch Lightning**: 简化深度学习训练流程

---

## 📞 联系方式

如有问题或建议，欢迎交流！

**项目路径**: `D:/GraduateDesign/Test/dcase2025_task6_baseline/ServerCodes/TestVersion/d25_t6`

**快速启动应用**:
```bash
cd d25_t6
python audio_caption_app.py
```

---

**项目完成日期**: 2026年2月14日

**最终状态**: ✅ 已完成，可投入使用

---

🎉 **恭喜完成项目！** 🎉


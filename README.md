# DCASE 2025 Task 6 - Audio-Text Retrieval

## 📁 项目结构

```
d25_t6/
├── docs/                    # 📚 所有项目文档（统一管理）
│   ├── QUICK_START.md      # 🔥 快速启动指南
│   ├── COMPLETE_FIX_GUIDE.md  # 完整修复指南
│   ├── CRITICAL_FIX_OVERFITTING.md  # 问题诊断
│   └── INDEX.md            # 文档索引
│
├── datasets/                # 数据集加载模块
├── train.py                 # 训练脚本
├── retrieval_module.py      # 模型定义
├── passt.py                 # PaSST音频编码器
└── predict.py               # 预测脚本

```

## 🚀 快速开始

### 1. 查看文档
所有文档都在 `docs/` 文件夹中：

```bash
cd docs
# 查看快速启动指南
cat QUICK_START.md
```

### 2. 启动训练
直接在项目根目录运行：

```bash
python train.py
```

### 3. 监控训练
打开 W&B 监控页面，关注 `val/mAP@10` 指标。

## 🎯 当前状态

### ✅ 已修复的问题
- **过拟合问题**: 训练到第10轮后指标下降
- **Hard Negative Mining**: 已禁用，避免过度拟合
- **正则化不足**: 已增加 weight_decay 和 dropout
- **缺少早停**: 已添加 EarlyStopping callback

### 🎯 训练目标
- **当前**: mAP@10 = 0.29 (过拟合前的峰值)
- **短期目标**: mAP@10 = 0.30 (Epoch 15)
- **中期目标**: mAP@10 = 0.32-0.33 (Epoch 25)
- **最终目标**: mAP@10 = 0.34-0.35 (Epoch 30-40)

## 📚 文档导航

### 🔥 必读文档
1. **[QUICK_START.md](docs/QUICK_START.md)** - 快速启动指南
2. **[COMPLETE_FIX_GUIDE.md](docs/COMPLETE_FIX_GUIDE.md)** - 完整修复指南
3. **[CRITICAL_FIX_OVERFITTING.md](docs/CRITICAL_FIX_OVERFITTING.md)** - 问题诊断

### 📖 参考文档
4. **[INDEX.md](docs/INDEX.md)** - 文档索引
5. **[LEARNING_RATE_RESTART_STRATEGY.md](docs/LEARNING_RATE_RESTART_STRATEGY.md)** - 学习率重启策略
6. **[RESTART_FROM_BEST_CHECKPOINT.md](docs/RESTART_FROM_BEST_CHECKPOINT.md)** - 检查点恢复

## 🔧 核心配置

### 模型
- **Audio Encoder**: PaSST (86M 参数)
- **Text Encoder**: RoBERTa-base (125M 参数)
- **总参数**: ~211M

### 数据集
- **Clotho**: ~3,800 样本
- **AudioCaps**: ~49,838 样本
- **总计**: ~53,638 样本

### 训练参数（已修复）
```python
# 损失函数
loss_type = "infonce"  # 禁用 Hard Negative Mining

# 正则化
weight_decay = 0.02    # 增加到 0.02
dropout_rate = 0.15    # 增加到 0.15

# 学习率
max_lr = 5e-5          # 降低到 5e-5
warmup_epochs = 5      # 增加到 5
restart_period = 10    # 每10轮重启

# 训练
max_epochs = 50        # 配合早停
batch_size = 48
accumulate_grad_batches = 2  # 有效batch=96

# 早停
patience = 8           # 8轮不提升就停止
```

## 📊 监控指标

### 主要指标
- **val/mAP@10**: 主要评估指标
- **val/R@5**: Recall@5
- **val/R@10**: Recall@10

### 关键检查点
| Epoch | 预期 mAP@10 | 说明 |
|-------|-------------|------|
| 5 | 0.24 | Warmup 完成 |
| 10 | 0.28 | 第一次重启，**不应该下降** ✅ |
| 15 | 0.30 | 突破 0.30 🎉 |
| 20 | 0.32 | 持续上升 |
| 30 | 0.34-0.35 | 达到目标 🎯 |

## 💡 常见问题

### Q: 如何查看文档？
**A**: 所有文档都在项目根目录的 `docs/` 文件夹中。

### Q: 训练多久能看到效果？
**A**: 前5轮是warmup，第10轮是关键检查点。如果第10轮后验证指标不下降，说明修复有效。

### Q: 如何判断是否还在过拟合？
**A**: 观察 `val/mAP@10` 曲线，如果在某个epoch后持续下降，说明仍在过拟合。

## 🚀 立即开始

```bash
# 启动训练
python train.py

# 打开 W&B 监控页面
# 观察 val/mAP@10 指标
```

## 📞 需要帮助？

1. 查看 `docs/QUICK_START.md` 快速启动指南
2. 查看 `docs/COMPLETE_FIX_GUIDE.md` 完整修复指南
3. 查看 `docs/INDEX.md` 文档索引

---

**目标**: 从 mAP@10 = 0.29 提升到 0.34-0.35 🎯

**状态**: ✅ 过拟合问题已修复，准备开始新训练

*最后更新: 2026-02-04*


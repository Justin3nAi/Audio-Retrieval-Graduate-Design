# 🎯 使用说明 - 自蒸馏 + MoE

## 📍 重要提示

### 项目结构
```
TestVersion/  (本地) 或 ProjectAR/  (服务器)
├── d25_t6/
│   ├── train.py
│   ├── retrieval_module.py
│   ├── self_distillation.py  ✅ 新增
│   ├── moe_module.py  ✅ 新增
│   ├── verify_installation.py  ✅ 新增
│   └── ...
└── ...
```

### 运行方式
**所有命令都在项目根目录（TestVersion或ProjectAR）运行**

使用 `python -m d25_t6.xxx` 格式，而不是 `cd d25_t6 && python xxx.py`

---

## ✅ 第1步: 验证安装

```bash
# 在项目根目录运行
python -m d25_t6.verify_installation
```

**预期输出**:
```
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
自蒸馏 + MoE 模块验证脚本
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

============================================================
测试1: 自蒸馏模块
============================================================
✅ 自蒸馏模块导入成功
✅ 蒸馏损失计算成功: 0.xxxx
✅ 损失形状正确: torch.Size([])

============================================================
测试2: MoE模块
============================================================
✅ SimpleMoEFFN导入成功
✅ 前向传播成功
...

============================================================
测试总结
============================================================
自蒸馏模块: ✅ 通过
MoE模块: ✅ 通过
retrieval_module集成: ✅ 通过
train.py参数: ✅ 通过

============================================================
总计: 4/4 测试通过
============================================================

🎉 所有测试通过！可以开始训练了！

推荐命令:
python -m d25_t6.train --use_self_distillation --distill_alpha 0.5 --roberta_base --logging
```

---

## 🚀 第2步: 启动训练

### 配置A: 仅自蒸馏（最推荐）⭐⭐⭐⭐⭐

```bash
# 在项目根目录运行
python -m d25_t6.train \
  --use_self_distillation \
  --distill_alpha 0.5 \
  --max_lr 4e-5 \
  --weight_decay 0.02 \
  --dropout_rate 0.15 \
  --roberta_base \
  --logging
```

**预期效果**: mAP@10 从 29.7% → 32-33% (+2-3%)

---

### 配置B: 自蒸馏 + MoE（激进）⭐⭐⭐⭐

```bash
# 在项目根目录运行
python -m d25_t6.train \
  --use_self_distillation \
  --distill_alpha 0.5 \
  --use_moe \
  --num_experts 4 \
  --top_k 2 \
  --use_cross_attention \
  --max_lr 4e-5 \
  --weight_decay 0.02 \
  --dropout_rate 0.15 \
  --roberta_base \
  --logging
```

**预期效果**: mAP@10 从 29.7% → 33-34% (+3-4%)

---

## 📊 第3步: 监控训练

### W&B Dashboard关键指标

| 指标 | 说明 | 期望值 |
|------|------|--------|
| `train/distill_loss` | 蒸馏损失 | 逐渐下降到0.2-0.3 |
| `train/contrastive_loss` | 对比损失 | 下降到1.6-1.8 |
| `val/mAP@10` | 验证mAP | Epoch 15 ≥ 31% |
| `train/moe_load_balance_loss` | MoE负载均衡 | 稳定在0.02左右 |

### 成功标准

#### Epoch 15检查点
- ✅ `val/mAP@10` ≥ 31%
- ✅ `train/distill_loss` < 0.4
- ✅ 没有过拟合（验证曲线持续上升）

#### Epoch 30最终目标
- ✅ `val/mAP@10` ≥ 32-33%（配置A）
- ✅ `val/mAP@10` ≥ 33-34%（配置B）
- ✅ `val/R@5` ≥ 48%
- ✅ `val/R@10` ≥ 62%

---

## ⚠️ 常见问题

### Q1: 运行验证脚本报错 "No module named 'd25_t6'"

**原因**: 不在项目根目录

**解决**:
```bash
# 确保在 TestVersion/ 或 ProjectAR/ 目录
pwd  # 检查当前目录

# 如果在 d25_t6/ 目录，返回上一级
cd ..

# 然后运行
python -m d25_t6.verify_installation
```

---

### Q2: 训练loss变成NaN

**原因**: 蒸馏损失权重太高

**解决**:
```bash
# 降低蒸馏权重
python -m d25_t6.train \
  --use_self_distillation \
  --distill_alpha 0.3 \
  --roberta_base \
  --logging
```

---

### Q3: 自蒸馏效果不明显

**原因**: 蒸馏损失权重太低

**解决**:
```bash
# 提高蒸馏权重
python -m d25_t6.train \
  --use_self_distillation \
  --distill_alpha 1.0 \
  --roberta_base \
  --logging
```

---

### Q4: MoE专家崩溃（只用1个专家）

**原因**: 负载均衡权重太低

**解决**:
```bash
# 提高负载均衡权重
python -m d25_t6.train \
  --use_self_distillation \
  --use_moe \
  --moe_load_balance_weight 0.05 \
  --roberta_base \
  --logging
```

---

## 📈 预期训练曲线

### 自蒸馏训练曲线

```
Epoch | train/distill_loss | val/mAP@10 | 状态
------|-------------------|------------|------
1     | 0.80              | 18.5%      | 正常
5     | 0.60              | 24.0%      | 正常
10    | 0.45              | 28.5%      | 正常
15    | 0.35              | 31.0%      | ✅ 突破30%
20    | 0.30              | 32.0%      | ✅ 达到目标
30    | 0.25              | 33.0%      | ✅ 超越目标
```

---

## 💡 关键优势

### 1. 无需下载额外模型 ✅
- 使用现有的EMA机制作为教师
- 只需PaSST和RoBERTa（你已有）
- 节省时间和存储空间

### 2. 风险低，稳定性高 ✅
- Kim论文验证有效（+4.3%）
- 实施简单，只添加一个损失项
- 不改变模型架构

### 3. 效果显著 ✅
- 预期提升2-4%
- 可能突破32%甚至34%
- 接近目标（35%）

---

## 📚 详细文档

- **README_DISTILLATION.md** - 主README
- **QUICK_START_DISTILLATION.md** - 快速参考
- **IMPLEMENTATION_SUMMARY.md** - 完整技术细节
- **COMPLETION_REPORT.md** - 完成报告
- **docs/SELF_DISTILLATION_MOE_GUIDE.md** - 深入指南

---

## ✅ 快速检查清单

开始训练前：

- [ ] 在项目根目录（TestVersion或ProjectAR）
- [ ] 运行 `python -m d25_t6.verify_installation`
- [ ] 确认 4/4 测试通过
- [ ] 准备监控W&B Dashboard
- [ ] 选择训练配置（推荐配置A）

---

## 🎯 推荐实施顺序

### 第1天: 验证和启动
```bash
# 1. 验证安装
python -m d25_t6.verify_installation

# 2. 启动训练（配置A）
python -m d25_t6.train \
  --use_self_distillation \
  --distill_alpha 0.5 \
  --roberta_base \
  --logging
```

### 第2天: 检查Epoch 15
- 如果 mAP@10 ≥ 31% → 继续训练
- 如果 mAP@10 < 31% → 调整参数

### 第3天: 检查Epoch 30
- 如果 mAP@10 ≥ 32% → 成功！尝试MoE
- 如果 mAP@10 < 32% → 调整参数

---

## 🚀 立即开始

```bash
# 在项目根目录（TestVersion或ProjectAR）运行
python -m d25_t6.verify_installation
python -m d25_t6.train --use_self_distillation --distill_alpha 0.5 --roberta_base --logging
```

**祝训练顺利！期待突破32%！** 🎉


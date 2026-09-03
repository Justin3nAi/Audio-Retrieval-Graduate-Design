# 🔒 项目路径规则

## 📍 唯一工作目录

**D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6**

---

## ⚠️ 重要规则

### 1. 所有代码修改必须在此路径下进行

- ✅ **正确路径**: `D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6\`
- ❌ **禁止使用**: `C:\Users\16285\.cursor\worktrees\d25_t6\*\`
- ❌ **禁止使用**: 任何其他worktree路径

### 2. 所有文档修改必须在此路径下进行

- ✅ **正确路径**: `D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6\docs\`
- ❌ **禁止使用**: 其他任何docs目录

### 3. 不再使用worktree

- ❌ 不再使用git worktree
- ✅ 直接在主项目路径下工作

---

## 📂 项目结构

```
D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6\
├── train.py                              ✅ 主训练脚本
├── retrieval_module.py                   ✅ 检索模型
├── clustering_classification.py          ✅ 聚类分类模块
├── online_distillation.py                ✅ 在线蒸馏模块
├── passt.py                              ✅ 音频编码器
├── moe_module.py                         ✅ MoE模块（未启用）
├── predict.py                            ✅ 推理脚本
├── datasets/                             ✅ 数据集加载
│   ├── audio_loading.py
│   ├── batch_collate.py
│   └── ...
└── docs/                                 ✅ 文档目录
    ├── INDEX.md                          📖 文档索引
    ├── TRAINING_PROGRESS.md              📖 训练进度（Epoch 11）
    ├── READY_TO_DEPLOY.md                📖 部署总结
    ├── DEPLOYMENT_CHECKLIST.md           📖 部署清单
    ├── STRUCTURAL_REFORM_PLAN.md         📖 技术方案
    ├── KIM_PAPER_ANALYSIS.md             📖 论文分析
    ├── BUG_FIXES.md                      📖 Bug修复
    ├── MEMORY_LEAK_FIX.md                📖 显存修复
    ├── USAGE_GUIDE.md                    📖 使用说明
    ├── OFFLINE_DEPENDENCIES.md           📖 依赖说明
    └── PROJECT_RULES.md                  📖 本文件
```

---

## 🎯 当前项目状态

### 版本信息
- **版本**: v6.0
- **核心技术**: 在线知识蒸馏 + 聚类引导分类
- **状态**: ✅ 训练中（Epoch 11/60）

### 已实现的功能

#### ✅ 在线知识蒸馏（Online Distillation）
- 使用EMA模型作为教师
- 无需预训练教师模型
- 文件：`online_distillation.py`
- 状态：✅ 已启用

#### ✅ 聚类引导分类（Clustering-Guided Classification）
- K-means聚类（50个类别）
- 辅助分类任务
- 文件：`clustering_classification.py`
- 状态：✅ 已启用

#### ❌ 多教师蒸馏（已放弃）
- 原计划：CLAP + AudioCLIP双教师
- 原因：效果不佳，语义空间不匹配
- 状态：❌ 已禁用

#### ❌ MoE（未启用）
- 原因：依赖交叉注意力（已禁用）
- 状态：❌ 未启用

### 当前训练配置

```python
# 核心参数
--batch_size 32                          # 降低到32避免OOM
--accumulate_grad_batches 2              # 有效batch=64
--max_lr 5e-5                            # 学习率
--restart_period 15                      # 学习率重启周期

# 启用的优化
--use_online_distillation True           # ✅ 在线蒸馏
--use_clustering_classification True     # ✅ 聚类分类
--use_attentive_aggregation True         # ✅ 注意力聚合
--use_attention_pooling True             # ✅ 注意力池化
--use_ema True                           # ✅ EMA

# 禁用的优化
--use_multi_teacher_distillation False   # ❌ 多教师蒸馏
--use_cross_attention False              # ❌ 交叉注意力
--use_moe False                          # ❌ MoE
```

### 当前训练进度

| 指标 | 当前值 | 目标 | 状态 |
|------|--------|------|------|
| Epoch | 11/60 | 60 | 🔄 训练中 |
| val/mAP@10 | 0.270 | 0.32-0.34 | 📈 上升中 |
| clustering_loss | 3.91 | 1.5 | ⚠️ 未下降 |
| online_distill_loss | 0.001 | 0.01-0.02 | ⚠️ 太小 |

---

## ✅ 已确认的最新文件

### 核心代码文件

1. ✅ **train.py**
   - 支持在线蒸馏和聚类分类
   - 显存优化（batch_size=32）
   - 学习率重启策略

2. ✅ **retrieval_module.py**
   - 集成在线蒸馏
   - 集成聚类分类
   - EMA教师模型

3. ✅ **clustering_classification.py**
   - K-means聚类（GPU加速）
   - 聚类分类器
   - 显存优化（训练后清理）

4. ✅ **online_distillation.py**
   - EMA教师模型
   - 在线蒸馏损失
   - 自动更新机制

### 文档文件

1. ✅ **TRAINING_PROGRESS.md** - 训练进度分析（Epoch 11）
2. ✅ **INDEX.md** - 文档索引
3. ✅ **READY_TO_DEPLOY.md** - 部署总结
4. ✅ **STRUCTURAL_REFORM_PLAN.md** - 技术方案
5. ✅ **BUG_FIXES.md** - Bug修复记录
6. ✅ **MEMORY_LEAK_FIX.md** - 显存泄露修复

---

## 🚨 操作检查清单

### 在进行任何修改前，必须确认：

- [ ] 当前工作路径是否为 `D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6\`
- [ ] 不是在worktree路径下工作
- [ ] 修改的文件路径正确
- [ ] 了解当前项目状态（使用聚类分类，不使用多教师蒸馏）

### 修改代码时：

```python
# ✅ 正确
file_path = "D:\\GraduateDesign\\Test\\dcase2025_task6_baseline\\ServerCodes\\TestVersion\\d25_t6\\train.py"

# ❌ 错误
file_path = "C:\\Users\\16285\\.cursor\\worktrees\\d25_t6\\*\\train.py"
```

### 修改文档时：

```python
# ✅ 正确
doc_path = "D:\\GraduateDesign\\Test\\dcase2025_task6_baseline\\ServerCodes\\TestVersion\\d25_t6\\docs\\INDEX.md"

# ❌ 错误
doc_path = "C:\\Users\\16285\\.cursor\\worktrees\\d25_t6\\*\\docs\\INDEX.md"
```

---

## 📋 快速验证命令

### 验证当前路径

```python
import os
current_path = os.getcwd()
correct_path = r"D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6"
assert current_path == correct_path, f"错误的路径: {current_path}"
```

### 验证文件存在

```python
import os
base_path = r"D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6"
assert os.path.exists(os.path.join(base_path, "train.py"))
assert os.path.exists(os.path.join(base_path, "clustering_classification.py"))
assert os.path.exists(os.path.join(base_path, "online_distillation.py"))
assert os.path.exists(os.path.join(base_path, "docs", "TRAINING_PROGRESS.md"))
```

---

## 🔐 规则总结

1. ✅ **唯一工作目录**: `D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6\`
2. ❌ **禁止使用worktree**: 不再使用任何worktree路径
3. ✅ **所有修改**: 必须在主项目路径下进行
4. ✅ **文件验证**: 修改前确认路径正确
5. ✅ **版本控制**: 保持所有文件为最新版本
6. ✅ **项目状态**: 使用聚类分类，暂时放弃多教师蒸馏

---

## 📊 技术方案演进

| 版本 | 方案 | 状态 | 效果 |
|------|------|------|------|
| v1.0 | 基线模型 | ✅ 完成 | mAP@10 = 0.29 |
| v2.0 | CLIP蒸馏 | ❌ 失败 | 语义不匹配 |
| v3.0 | 双教师蒸馏 | ❌ 放弃 | 效果不佳 |
| **v6.0** | **在线蒸馏+聚类** | **🔄 训练中** | **预期0.32-0.34** |

---

**此规则文件更新于**: 2026-02-08  
**目的**: 确保所有开发工作在正确的项目路径下进行，并反映当前实际项目状态  
**重要性**: ⭐⭐⭐⭐⭐ 必须严格遵守  
**当前状态**: 使用聚类分类，暂时放弃多教师蒸馏

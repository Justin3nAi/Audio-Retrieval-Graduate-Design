# 📚 项目文档索引

本文件夹包含所有项目相关的文档和指南。

---

## 📁 文档列表

### 🔧 训练问题诊断与修复

1. **CRITICAL_FIX.md** - 紧急修复：Loss反弹问题
   - 问题：交叉注意力导致训练崩溃
   - 解决：禁用交叉注意力，修复autocast警告
   - 适用场景：训练loss突然上升

2. **TRAINING_DIAGNOSIS.md** - 训练效果不佳的诊断与修复
   - 问题：mAP@10只有26%
   - 解决：简化投影头，降低模型复杂度
   - 适用场景：训练效果不理想

### 📥 数据集配置指南

3. **AUDIOCAPS_SETUP.md** - AudioCaps自动下载指南
   - 自动下载和配置AudioCaps数据集
   - 包含三种下载方法
   - 适用场景：有外网访问权限

4. **AUDIOCAPS_MANUAL_SETUP.md** - AudioCaps手动下载指南（中国境内）
   - 详细的手动下载和上传步骤
   - 适用于无法访问外网的服务器
   - 包含完整的故障排除

5. **AUDIOCAPS_READY.md** - AudioCaps配置完成检查清单
   - 验证数据集是否正确配置
   - 包含快速测试命令
   - 适用场景：配置完成后验证

6. **VERIFY_AUDIOCAPS.md** - AudioCaps加载验证指南
   - 详细的日志输出说明
   - 如何判断数据集是否真的加载成功
   - 包含成功和失败的标志

---

## 🗂️ 文档分类

### 按问题类型

| 问题类型 | 相关文档 |
|---------|---------|
| 训练不稳定 | CRITICAL_FIX.md |
| 性能不佳 | TRAINING_DIAGNOSIS.md |
| 数据集配置 | AUDIOCAPS_SETUP.md, AUDIOCAPS_MANUAL_SETUP.md |
| 验证测试 | AUDIOCAPS_READY.md, VERIFY_AUDIOCAPS.md |

### 按使用场景

| 场景 | 推荐文档 |
|------|---------|
| 首次配置 | AUDIOCAPS_MANUAL_SETUP.md → AUDIOCAPS_READY.md |
| 训练出问题 | CRITICAL_FIX.md → TRAINING_DIAGNOSIS.md |
| 验证数据集 | VERIFY_AUDIOCAPS.md |
| 性能优化 | TRAINING_DIAGNOSIS.md |

---

## 🚀 快速导航

### 我遇到了训练问题
1. Loss突然上升？→ [CRITICAL_FIX.md](CRITICAL_FIX.md)
2. mAP太低？→ [TRAINING_DIAGNOSIS.md](TRAINING_DIAGNOSIS.md)

### 我要配置AudioCaps
1. 有外网？→ [AUDIOCAPS_SETUP.md](AUDIOCAPS_SETUP.md)
2. 无外网（中国境内）？→ [AUDIOCAPS_MANUAL_SETUP.md](AUDIOCAPS_MANUAL_SETUP.md)
3. 配置完成？→ [AUDIOCAPS_READY.md](AUDIOCAPS_READY.md)
4. 验证加载？→ [VERIFY_AUDIOCAPS.md](VERIFY_AUDIOCAPS.md)

---

## 📊 文档更新记录

| 日期 | 文档 | 更新内容 |
|------|------|---------|
| 2026-02-01 | 所有文档 | 创建并整理到docs文件夹 |

---

## 💡 使用建议

1. **按顺序阅读**：如果是新手，建议按照文档编号顺序阅读
2. **问题导向**：如果遇到具体问题，直接查看对应的文档
3. **保持更新**：所有新的文档都会添加到这个文件夹中

---

## 📞 需要帮助？

如果文档中没有解决你的问题，请：
1. 检查是否有新的文档更新
2. 查看训练日志中的错误信息
3. 参考多个相关文档

---

**最后更新：2026-02-01**


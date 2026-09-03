# 候选Caption提取指南

## 📝 概述

本指南说明如何从训练数据集中提取所有真实的caption，用于扩展应用的识别能力。

## 🎯 为什么需要提取Caption？

**当前问题**：
- 应用只能从70+个预定义描述中选择
- 覆盖范围有限，灵活性不足

**解决方案**：
- 从Clotho和AudioCaps数据集提取所有真实caption
- 预计可获得**数万个**唯一的音频描述
- 大幅提升识别的多样性和准确性

## 🚀 使用步骤

### 步骤1：提取Caption

在Anaconda Prompt中运行：

```bash
# 激活环境
conda activate d25_t6

# 切换到audio_app目录
D:
cd D:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6\audio_app

# 运行提取脚本
python extract_captions.py --data_path ../data
```

**参数说明**：
- `--data_path`: 数据集根目录（包含Clotho和AudioCaps）
- `--output`: 输出文件名（默认：candidate_captions.json）

### 步骤2：验证输出

脚本会生成 `candidate_captions.json` 文件，包含所有唯一的caption。

检查文件：
```bash
# 查看文件大小
dir candidate_captions.json

# 查看前几行（可选）
type candidate_captions.json | more
```

### 步骤3：启动应用

应用会自动加载新的caption库：

```bash
python run_app.py
```

或者明确指定caption文件：

```bash
python audio_caption_app.py --caption_file candidate_captions.json
```

## 📊 预期结果

根据数据集规模，预计可以提取：

- **Clotho**：
  - dev: ~3,800个音频 × 5个caption = ~19,000个caption
  - val: ~1,000个音频 × 5个caption = ~5,000个caption
  - eval: ~1,000个音频 × 5个caption = ~5,000个caption

- **AudioCaps**：
  - train: ~49,000个音频 × 1个caption = ~49,000个caption
  - val: ~500个音频 × 1个caption = ~500个caption
  - test: ~1,000个音频 × 1个caption = ~1,000个caption

**去重后预计**：**20,000 - 40,000** 个唯一caption

## 🔍 提取过程说明

脚本会：
1. 遍历所有数据集子集
2. 提取每个样本的caption
3. 转换为小写并去除首尾空格
4. 使用set自动去重
5. 排序后保存为JSON格式

## ⚠️ 注意事项

1. **数据集路径**：确保Clotho和AudioCaps数据集已正确解压到data目录
2. **内存占用**：提取过程可能需要几分钟，取决于数据集大小
3. **文件编码**：输出文件使用UTF-8编码，支持多语言
4. **备用方案**：如果提取失败，应用会自动回退到预定义的70+个描述

## 🎨 自定义Caption库

你也可以手动编辑 `candidate_captions.json` 文件：

```json
[
  "a dog barking loudly",
  "birds chirping in the morning",
  "rain falling on the roof",
  ...
]
```

**格式要求**：
- JSON数组格式
- 每个caption为字符串
- 建议使用小写
- UTF-8编码

## 🔄 更新Caption库

如果数据集有更新，重新运行提取脚本即可：

```bash
python extract_captions.py --data_path ../data --output candidate_captions_v2.json
```

然后启动应用时指定新文件：

```bash
python audio_caption_app.py --caption_file candidate_captions_v2.json
```

## 💡 性能优化建议

如果caption数量过多（>50,000），可以考虑：

1. **采样策略**：只使用训练集的caption
2. **频率过滤**：移除出现次数少的caption
3. **语义聚类**：合并相似的caption
4. **缓存embeddings**：预计算并保存caption embeddings到文件

## 📞 问题排查

**问题1：数据集加载失败**
```
解决：检查data_path是否正确，确保包含CLOTHO和AUDIOCAPS文件夹
```

**问题2：提取的caption数量很少**
```
解决：检查数据集是否完整下载，查看脚本输出的错误信息
```

**问题3：应用启动时未加载caption文件**
```
解决：确保candidate_captions.json在audio_app目录下，或使用--caption_file参数指定路径
```

## ✅ 完成检查清单

- [ ] 数据集已正确解压到data目录
- [ ] 运行extract_captions.py脚本
- [ ] 生成了candidate_captions.json文件
- [ ] 文件大小合理（通常几百KB到几MB）
- [ ] 重启应用并验证加载了新的caption库
- [ ] 测试音频识别，观察结果多样性

---

**提示**：使用真实数据集的caption可以显著提升应用的实用性和准确性！

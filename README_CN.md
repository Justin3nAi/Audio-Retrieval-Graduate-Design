# 🎵 多编码器音频-文本检索系统

一个基于检索的音频文本匹配系统，结合PaSST和CLAP编码器与注意力融合机制，在Clotho评估集上达到 **mAP@10 = 0.32**。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 简体中文

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design.git
cd Audio-Retrieval-Graduate-Design
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载模型权重

**方式A：从HuggingFace自动下载（推荐）**
```bash
python scripts/download_model.py
```

**方式B：手动下载**
- 从 [HuggingFace Hub](https://huggingface.co/Justinzhu09/Audio-Retrieval-Graduate-Design) 下载
- 将 `model.ckpt` 放到 `audio_app/checkpoints/` 目录

### 4. 运行Gradio Web应用
```bash
# Linux/Mac
bash audio_app/start_app.sh

# Windows
audio_app\start_app.bat

# 或直接运行
python audio_app/run_app.py
```

应用将在 `http://localhost:7860` 启动

---

## 📖 项目简介

本项目实现了一个**基于检索的音频文本匹配系统**，主要功能包括：

1. **音频编码**：使用互补的编码器（PaSST + CLAP）
2. **表征融合**：基于注意力机制的融合策略
3. **描述检索**：从80,045个候选描述库中检索
4. **Web部署**：CPU上2-3秒完成处理的实时应用

### 主要特性

- ✅ **多编码器融合**：结合时频模式（PaSST）和音频文本语义（CLAP）
- ✅ **基于检索**：快速相似度搜索，无生成开销
- ✅ **实时推理**：标准CPU上每个音频片段2-3秒
- ✅ **Web界面**：易于使用的Gradio应用

---

## 🎯 性能表现

| 模型 | mAP@10 | R@1 | R@10 |
|-------|--------|-----|------|
| PaSST单编码器 | 0.290 | 0.181 | 0.588 |
| CLAP单编码器 | 0.271 | 0.161 | 0.560 |
| **多编码器融合** | **0.325** | **0.203** | **0.623** |

- **相比PaSST提升12.1%**
- **相比CLAP提升20.0%**
- 在Clotho数据集上评估

---

## 📁 项目结构

```
Audio-Retrieval-Graduate-Design/
├── audio_app/                          # Gradio web应用
│   ├── audio_caption_app.py           # 主应用程序
│   ├── run_app.py                     # 应用启动器
│   ├── candidate_captions.json        # 80,045个候选描述
│   ├── requirements_app.txt           # 应用依赖
│   ├── start_app.sh                   # Linux/Mac启动脚本
│   └── start_app.bat                  # Windows启动脚本
│
├── retrieval_module.py                # 核心检索模型
├── multi_audio_encoder.py             # PaSST + CLAP融合
├── passt.py                           # PaSST编码器封装
├── train.py                           # 训练脚本
├── predict.py                         # 推理脚本
│
├── datasets/                          # 数据加载工具
│   ├── audio_loading.py
│   └── __init__.py
│
├── scripts/                           # 辅助脚本
│   ├── download_model.py              # 模型权重下载器
│   └── extract_captions.py            # 描述提取工具
│
├── requirements.txt                   # 项目依赖
├── .gitignore
└── README.md
```

---

## 💻 使用方法

### Web应用（推荐）

1. 启动Gradio界面：
```bash
python audio_app/run_app.py
```

2. 在浏览器打开 `http://localhost:7860`

3. 上传音频文件或通过麦克风录音

4. 获取前10个检索描述及相似度分数

### 命令行推理

```bash
python predict.py --audio_path path/to/audio.wav --checkpoint path/to/model.ckpt
```

### 训练

```bash
python train.py \
    --batch_size 32 \
    --max_epochs 50 \
    --learning_rate 2e-5
```

---

## 🔧 系统要求

### 最低要求
- Python 3.8+
- 4GB RAM
- 支持纯CPU运行

### 推荐配置
- Python 3.8+
- 8GB+ RAM
- NVIDIA GPU 12GB+ VRAM（用于训练）

### 依赖项
- PyTorch 2.0+
- Transformers 4.30+
- Gradio 4.0+
- librosa 0.10+
- hear21passt (PaSST模型)
- aac-datasets (CLAP模型)

---

## 📊 技术细节

### 系统架构

**音频分支：**
- PaSST：在AudioSet上预训练，捕获时频模式
- CLAP：通过音频文本对比学习预训练
- 注意力融合：动态加权编码器输出

**文本分支：**
- RoBERTa-large：1024维文本嵌入
- 候选库：从Clotho + AudioCaps提取的80,045个唯一描述

**检索：**
- 在联合嵌入空间中计算余弦相似度
- 返回top-K最相似描述

### 训练配置

- **损失函数**: InfoNCE对比损失
- **优化器**: AdamW (lr=2e-5)
- **数据增强**: SpecAugment（时间掩码15帧，频率掩码2个bins）
- **批大小**: 32
- **训练轮数**: 50

---

## 📝 引用

如果你在研究中使用了本代码，请引用：

```bibtex
@misc{audio-text-retrieval-2026,
  author = {Yuanzhi Zhu},
  title = {Multi-Encoder Audio-Text Retrieval System},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design}
}
```

---

## 🙏 致谢

本项目基于以下开源工作：

- [PaSST](https://github.com/kkoutini/PaSST) - Patchout频谱图Transformer
- [CLAP](https://github.com/LAION-AI/CLAP) - 对比语言-音频预训练
- [Clotho](https://zenodo.org/record/3490684) - 音频描述数据集
- [AudioCaps](https://audiocaps.github.io/) - 音频描述数据集

---

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🐛 问题和贡献

- **Bug报告**: [提交Issue](https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design/issues)
- **功能请求**: [提交Issue](https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design/issues)
- **Pull Requests**: 欢迎！

---

## 📧 联系方式

如有问题或讨论：

- **GitHub Issues**: [项目Issues](https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design/issues)
- **Email**: zhuyuanzhi09@126.com

---

## 🌟 Star历史

如果觉得这个项目对你有帮助，请给个⭐!

---

**最后更新**: 2026年3月  
**状态**: 活跃开发中

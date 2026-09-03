# 🎵 音频内容识别应用使用指南

## 📖 简介

这是一个基于训练好的音频-文本检索模型（mAP@10=0.32）构建的Web应用，用户可以上传音频文件，系统会自动识别音频内容并返回文字描述。

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /d:/GraduateDesign/Test/dcase2025_task6_baseline/ServerCodes/TestVersion/d25_t6

# 安装应用所需依赖
pip install -r requirements_app.txt
```

### 2. 启动应用

```bash
# 基本启动（本地访问）
python audio_caption_app.py

# 指定checkpoint路径
python audio_caption_app.py --checkpoint Train4(0.32)/mAP@10=0.32.ckpt

# 指定端口
python audio_caption_app.py --port 8080

# 创建公网链接（可以分享给他人）
python audio_caption_app.py --share
```

### 3. 访问应用

启动成功后，在浏览器中访问：
```
http://localhost:7860
```

如果使用了`--share`参数，会生成一个公网链接（有效期72小时）：
```
https://xxxxx.gradio.live
```

## 🎯 功能特性

### 核心功能

1. **音频上传**
   - 支持拖拽上传
   - 支持点击选择文件
   - 支持麦克风录音

2. **智能识别**
   - 自动识别音频内容
   - 返回Top-K最匹配的描述
   - 显示置信度评分

3. **结果展示**
   - 音频时长信息
   - 按相似度排序的描述列表
   - 置信度可视化（颜色标识）

### 支持的音频格式

- WAV
- MP3
- FLAC
- OGG
- M4A
- 其他librosa支持的格式

### 识别类别（70+种）

应用可以识别以下类型的音频内容：

**自然环境**：
- 鸟鸣、雨声、海浪、风声、雷声、流水、瀑布等

**城市环境**：
- 汽车、交通、火车、飞机、施工、人群、脚步声、门声等

**室内声音**：
- 键盘、电话、时钟、吸尘器、洗衣机、微波炉等

**动物声音**：
- 狗叫、猫叫、马嘶、牛叫、鸡鸣、昆虫等

**音乐乐器**：
- 钢琴、吉他、鼓、小提琴、长笛、歌声、管弦乐等

**人类活动**：
- 婴儿哭声、儿童玩耍、笑声、咳嗽、打喷嚏、鼓掌等

**工具机械**：
- 锤子、锯子、电钻、电锯、割草机等

**厨房声音**：
- 水煮沸、煎炸、切菜、搅拌机、餐具碰撞等

**其他**：
- 火焰、玻璃破碎、纸张、拉链、硬币、铃声、口哨等

## 💡 使用技巧

### 获得最佳识别效果

1. **音频质量**
   - 使用清晰的音频文件
   - 避免过多背景噪音
   - 建议采样率 ≥16kHz

2. **音频时长**
   - 推荐时长：1-30秒
   - 过短可能信息不足
   - 过长会自动截取前30秒

3. **内容单一**
   - 单一声音源识别效果最好
   - 混合声音可能降低置信度

### 理解置信度

- 🟢 **≥80%**：高度匹配，结果可信
- 🟡 **60-80%**：较好匹配，可能正确
- 🔴 **<60%**：低置信度，仅供参考

## 🔧 高级配置

### 自定义候选描述

如果想添加更多识别类别，编辑`audio_caption_app.py`中的`candidate_captions`列表：

```python
self.candidate_captions = [
    "your custom caption 1",
    "your custom caption 2",
    # ... 更多描述
]
```

### 调整返回结果数量

在Web界面中使用滑块调整，或在代码中修改默认值：

```python
top_k_slider = gr.Slider(
    minimum=1,
    maximum=10,
    value=5,  # 修改这里的默认值
    step=1,
    label="返回结果数量"
)
```

## 📊 技术细节

### 模型架构

- **音频编码器**：PaSST + CLAP 双编码器融合
- **文本编码器**：RoBERTa-base
- **嵌入维度**：1024
- **训练数据**：Clotho + AudioCaps (~68K样本)

### 工作流程

1. **音频预处理**
   - 重采样到32kHz
   - 填充/截断到最大30秒
   - 转换为张量

2. **特征提取**
   - 通过音频编码器提取embedding
   - 通过文本编码器提取候选描述embedding

3. **相似度计算**
   - 使用余弦相似度
   - 归一化后计算内积
   - 排序返回Top-K结果

### 性能指标

- **mAP@10**: 0.32
- **推理速度**: ~1-2秒/音频（GPU）
- **显存占用**: ~2-3GB（GPU）

## 🐛 常见问题

### Q1: 启动时提示找不到checkpoint

**解决方案**：
```bash
# 确保checkpoint文件存在
ls Train4\(0.32\)/mAP@10=0.32.ckpt

# 或指定完整路径
python audio_caption_app.py --checkpoint "完整路径/mAP@10=0.32.ckpt"
```

### Q2: 识别结果不准确

**可能原因**：
- 音频质量差或噪音多
- 音频内容不在训练类别中
- 音频时长过短

**改进方法**：
- 使用更清晰的音频
- 添加更多候选描述
- 使用更长的音频片段

### Q3: 启动慢或推理慢

**优化方法**：
- 使用GPU（自动检测）
- 减少候选描述数量
- 降低音频采样率

### Q4: 端口被占用

**解决方案**：
```bash
# 使用其他端口
python audio_caption_app.py --port 8080
```

## 🌐 部署到服务器

### 本地网络部署

```bash
# 启动应用，允许局域网访问
python audio_caption_app.py --port 7860

# 其他设备访问
# http://服务器IP:7860
```

### 公网部署（使用Gradio Share）

```bash
# 创建临时公网链接（72小时有效）
python audio_caption_app.py --share

# 会生成类似这样的链接：
# https://xxxxx.gradio.live
```

### 生产环境部署

如果需要长期稳定的服务，建议：

1. **使用反向代理**（Nginx）
2. **使用进程管理器**（systemd, supervisor）
3. **配置HTTPS**
4. **添加认证机制**

示例systemd服务配置：

```ini
[Unit]
Description=Audio Caption App
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/d25_t6
ExecStart=/usr/bin/python3 audio_caption_app.py --port 7860
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📝 更新日志

### v1.0.0 (2026-02-14)

- ✅ 初始版本发布
- ✅ 支持音频上传和麦克风录音
- ✅ 70+种音频类别识别
- ✅ 美观的Web界面
- ✅ 置信度可视化

## 🎉 总结

这个应用将你训练的音频-文本检索模型转化为了一个实用的工具，可以：

- ✅ 快速识别音频内容
- ✅ 提供直观的Web界面
- ✅ 支持多种音频格式
- ✅ 易于部署和分享

**立即开始使用**：
```bash
python audio_caption_app.py
```

然后在浏览器中打开 http://localhost:7860 即可！


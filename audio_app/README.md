# 🎵 音频内容识别应用

基于深度学习的音频识别Web应用，上传音频即可获得AI生成的文字描述。

## 🚀 快速开始

### Windows用户（最简单）
```bash
双击运行 start_app.bat
```

### 手动启动
```bash
# 1. 安装依赖（首次运行）
pip install -r requirements_app.txt

# 2. 启动应用
python audio_caption_app.py

# 3. 浏览器访问
# http://localhost:7860
```

## ✨ 功能特性

- 🎯 识别70+种音频场景（鸟鸣、雨声、汽车、钢琴等）
- 📊 显示置信度评分
- 🎤 支持文件上传和麦克风录音
- 🌐 美观的Web界面
- ⚡ 快速响应（1-2秒）

## 📁 文件说明

```
audio_app/
├── audio_caption_app.py       # 主应用程序
├── test_app.py                # 环境测试脚本
├── start_app.bat              # Windows启动脚本
├── start_app.sh               # Linux/Mac启动脚本
├── requirements_app.txt       # 依赖列表
├── README_APP.md              # 本文件
├── Train4(0.32)/              # 模型文件夹
│   └── mAP@10=0.32.ckpt      # 训练好的模型
└── docs/                      # 详细文档
    ├── APP_USAGE.md           # 完整使用指南
    └── PROJECT_SUMMARY.md     # 项目总结
```

## 🧪 测试环境

启动前可以先测试环境：

```bash
python test_app.py
```

会检查：
- ✅ Python版本和依赖包
- ✅ 模型文件
- ✅ GPU/CUDA支持
- ✅ 模型加载

## 💡 使用技巧

### 获得最佳效果
- 使用清晰的音频（避免噪音）
- 音频时长1-30秒最佳
- 单一声音源效果更好

### 置信度说明
- 🟢 ≥80%: 高度匹配
- 🟡 60-80%: 较好匹配
- 🔴 <60%: 低置信度

## 🔧 高级选项

### 指定端口
```bash
python audio_caption_app.py --port 8080
```

### 创建公网链接（可分享）
```bash
python audio_caption_app.py --share
```

### 自定义模型路径
```bash
python audio_caption_app.py --checkpoint "你的模型路径.ckpt"
```

## 📊 技术信息

- **模型性能**: mAP@10 = 0.32
- **架构**: PaSST + CLAP 双编码器融合
- **训练数据**: Clotho + AudioCaps (~68K样本)
- **推理速度**: ~1-2秒/音频（GPU）

## 🎯 支持的音频类别

- 🌳 自然环境: 鸟鸣、雨声、海浪、风声、雷声
- 🚗 城市环境: 汽车、交通、火车、飞机、人群
- 🏠 室内声音: 键盘、电话、吸尘器、洗衣机
- 🐕 动物声音: 狗叫、猫叫、马嘶、鸡鸣
- 🎹 音乐乐器: 钢琴、吉他、鼓、小提琴
- 👶 人类活动: 婴儿哭声、笑声、咳嗽、鼓掌
- 🔨 工具机械: 锤子、电钻、电锯、割草机
- 🍳 厨房声音: 煎炸、切菜、搅拌、餐具碰撞

## 🐛 常见问题

**Q: 提示找不到模块？**  
A: 运行 `pip install -r requirements_app.txt`

**Q: 找不到checkpoint文件？**  
A: 确保 `Train4(0.32)/mAP@10=0.32.ckpt` 存在

**Q: 识别不准确？**  
A: 检查音频质量，确保内容在支持的类别中

**Q: 端口被占用？**  
A: 使用 `--port` 参数指定其他端口

## 📚 详细文档

- [完整使用指南](docs/APP_USAGE.md)
- [项目总结报告](docs/PROJECT_SUMMARY.md)

## 🎉 立即开始

```bash
# 测试环境
python test_app.py

# 启动应用
python audio_caption_app.py

# 浏览器访问
# http://localhost:7860
```

---

**模型版本**: mAP@10=0.32  
**更新日期**: 2026-02-14  
**状态**: ✅ 可用




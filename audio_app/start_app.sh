#!/bin/bash

echo "========================================"
echo "🎵 音频内容识别系统"
echo "========================================"
echo ""

cd "$(dirname "$0")"

echo "📋 检查依赖..."
python -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  检测到缺少依赖，正在安装..."
    pip install -r requirements_app.txt
    echo ""
fi

echo "🚀 启动应用..."
echo ""
echo "💡 提示："
echo "   - 本地访问: http://localhost:7860"
echo "   - 按 Ctrl+C 停止服务"
echo "   - 应用目录: $(pwd)"
echo ""
echo "========================================"
echo ""

python audio_caption_app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 启动失败！请检查："
    echo "   1. 是否安装了所有依赖？运行: pip install -r requirements_app.txt"
    echo "   2. 模型文件是否存在？检查: Train4(0.32)/mAP@10=0.32.ckpt"
    echo "   3. 查看详细错误信息"
    echo ""
fi


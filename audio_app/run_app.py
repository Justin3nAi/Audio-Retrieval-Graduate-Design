"""
应用启动包装脚本
解决模块导入路径问题
"""

import os
import sys

# 设置环境变量强制使用英文
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LANGUAGE'] = 'en_US:en'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# 获取脚本所在目录（audio_app）
script_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父目录（d25_t6）
parent_dir = os.path.dirname(script_dir)

# 获取祖父目录（包含d25_t6的目录）
grandparent_dir = os.path.dirname(parent_dir)

# 添加祖父目录到Python路径，这样可以 import d25_t6
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# 切换工作目录到audio_app
os.chdir(script_dir)

print(f"📁 工作目录: {os.getcwd()}")
print(f"📦 Python路径已添加: {grandparent_dir}")

# 现在导入并运行主应用
if __name__ == "__main__":
    # 导入主应用模块
    from audio_caption_app import create_app
    import argparse
    
    parser = argparse.ArgumentParser(description="音频描述生成应用")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='Train4(0.32)/mAP@10=0.32.ckpt',
        help='模型checkpoint路径'
    )
    parser.add_argument(
        '--caption_file',
        type=str,
        default='candidate_captions.json',
        help='候选caption文件路径'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Web服务端口'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='是否创建公网链接'
    )
    
    args = parser.parse_args()
    
    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误: 找不到checkpoint文件: {args.checkpoint}")
        print(f"   当前工作目录: {os.getcwd()}")
        exit(1)
    
    # 检查caption文件
    caption_file_path = args.caption_file if os.path.exists(args.caption_file) else None
    if caption_file_path:
        print(f"✅ 找到候选caption文件: {caption_file_path}")
    else:
        print(f"⚠️ 未找到候选caption文件: {args.caption_file}")
        print(f"   将使用预定义的候选描述列表")
    
    # 创建并启动应用
    print("\n" + "="*60)
    print("🚀 启动音频内容识别系统...")
    print("="*60)
    
    app = create_app(args.checkpoint, caption_file_path)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )





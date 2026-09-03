"""
快速测试脚本 - 验证应用是否能正常工作
"""

import os
import sys

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    print("-" * 50)
    
    # 检查Python版本
    import sys
    print(f"✅ Python版本: {sys.version.split()[0]}")
    
    # 检查必要的包
    required_packages = [
        'torch',
        'gradio', 
        'librosa',
        'numpy',
        'pandas',
        'transformers',
        'lightning'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行: pip install -r requirements_app.txt")
        return False
    
    print("-" * 50)
    return True


def check_checkpoint():
    """检查模型文件"""
    print("\n🔍 检查模型文件...")
    print("-" * 50)
    
    checkpoint_path = "Train4(0.32)/mAP@10=0.32.ckpt"
    
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"✅ 模型文件存在: {checkpoint_path}")
        print(f"   文件大小: {size_mb:.2f} MB")
        print("-" * 50)
        return True
    else:
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print(f"   当前目录: {os.getcwd()}")
        print("\n请确保模型文件在正确的位置！")
        print("-" * 50)
        return False


def check_cuda():
    """检查CUDA可用性"""
    print("\n🔍 检查GPU支持...")
    print("-" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            print("   (CPU模式下推理速度会较慢)")
    except Exception as e:
        print(f"❌ 检查CUDA时出错: {e}")
    
    print("-" * 50)


def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")
    print("-" * 50)
    
    try:
        import sys
        import os
        # 添加父目录到路径，以便导入d25_t6模块
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from d25_t6.retrieval_module import AudioRetrievalModel
        import torch
        
        checkpoint_path = "Train4(0.32)/mAP@10=0.32.ckpt"
        
        print("正在加载模型...")
        model = AudioRetrievalModel.load_from_checkpoint(checkpoint_path)
        print("✅ 模型加载成功！")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   总参数量: {total_params / 1e6:.2f}M")
        
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("-" * 50)
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 50)
    print("🧪 音频识别应用 - 环境测试")
    print("=" * 50 + "\n")
    
    # 1. 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败！请先安装依赖。")
        return False
    
    # 2. 检查模型文件
    if not check_checkpoint():
        print("\n❌ 模型文件检查失败！")
        return False
    
    # 3. 检查CUDA
    check_cuda()
    
    # 4. 测试模型加载
    if not test_model_loading():
        print("\n❌ 模型加载测试失败！")
        return False
    
    # 全部通过
    print("\n" + "=" * 50)
    print("✅ 所有检查通过！")
    print("=" * 50)
    print("\n🚀 可以启动应用了：")
    print("   python audio_caption_app.py")
    print("\n或者使用快捷脚本：")
    print("   Windows: start_app.bat")
    print("   Linux/Mac: ./start_app.sh")
    print("\n" + "=" * 50 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


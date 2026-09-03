"""

BEATs模型简化加载器

不依赖fairseq，直接加载BEATs模型

"""



import torch

import torch.nn as nn





def load_beats_model_simple(model_path, device='cuda'):

    """

    简化的BEATs模型加载（不依赖fairseq）

    

    Args:

        model_path: BEATs模型路径

        device: 设备

    

    Returns:

        beats_wrapper: BEATs包装器

    """

    try:

        print(f"📥 加载BEATs模型（简化版）: {model_path}")

        

        # 加载checkpoint

        checkpoint = torch.load(model_path, map_location='cpu')

        

        # 检查checkpoint内容

        if 'model' not in checkpoint or 'cfg' not in checkpoint:

            print("❌ BEATs模型文件格式不正确")

            return None

        

        # 尝试导入BEATs

        import sys

        beats_lib_path = '/root/autodl-tmp/ProjectAR/beats_lib'

        if beats_lib_path not in sys.path:

            sys.path.insert(0, beats_lib_path)

        

        try:

            from BEATs import BEATs, BEATsConfig

        except ImportError as e:

            print(f"❌ 无法导入BEATs: {e}")

            print("   BEATs.py可能缺少某些依赖")

            return None

        

        # 创建模型

        cfg = BEATsConfig(checkpoint['cfg'])

        beats_model = BEATs(cfg)

        beats_model.load_state_dict(checkpoint['model'])

        beats_model.eval()

        beats_model.to(device)

        

        # 冻结参数

        for param in beats_model.parameters():

            param.requires_grad = False

        

        # 创建包装器

        class BEATsWrapper(nn.Module):

            def __init__(self, model):

                super().__init__()

                self.model = model

            

            def forward(self, audio):

                """

                前向传播（兼容接口）

                

                Args:

                    audio: (batch, channels, samples) 或 (batch, samples)

                

                Returns:

                    features: (batch, 768)

                """

                return self.extract_features(audio)

            

            def extract_features(self, audio):

                """

                提取音频特征

                

                Args:

                    audio: (batch, channels, samples) 或 (batch, samples)

                

                Returns:

                    features: (batch, 768)

                """

                # 确保是单声道

                if audio.dim() == 3:

                    audio = audio.mean(dim=1)  # (batch, samples)

                

                # BEATs期望的输入格式

                with torch.no_grad():

                    # 填充到合适的长度（BEATs通常需要固定长度）

                    # 这里假设输入是10秒@16kHz = 160000 samples

                    target_length = 160000

                    if audio.shape[-1] < target_length:

                        # 填充

                        padding = target_length - audio.shape[-1]

                        audio = torch.nn.functional.pad(audio, (0, padding))

                    elif audio.shape[-1] > target_length:

                        # 截断

                        audio = audio[:, :target_length]

                    

                    # 提取特征

                    features = self.model.extract_features(audio)[0]

                

                return features

        

        wrapper = BEATsWrapper(beats_model)

        

        print(f"✅ BEATs模型加载成功（简化版）")

        return wrapper

        

    except Exception as e:

        print(f"❌ BEATs加载失败: {e}")

        import traceback

        traceback.print_exc()

        return None





if __name__ == '__main__':

    # 测试

    model_path = '/root/autodl-tmp/teacher_models/beats/BEATs_iter3_plus_AS2M.pt'

    beats_model = load_beats_model_simple(model_path)

    

    if beats_model is not None:

        # 测试前向传播

        dummy_audio = torch.randn(2, 2, 320000)  # (batch=2, channels=2, samples)

        features = beats_model.extract_features(dummy_audio)

        print(f"✅ 特征提取成功: {features.shape}")




















































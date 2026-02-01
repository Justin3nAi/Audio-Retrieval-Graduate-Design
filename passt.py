import torch
import torch.nn as nn
import os
from hear21passt.base import get_model_passt, AugmentMelSTFT

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="hearpasst")
warnings.filterwarnings("ignore", category=UserWarning, module="hearpasst")


class PaSSTSNoOverlapWrapper(torch.nn.Module):
    def __init__(self, s_patchout_t=15, s_patchout_f=2, pretrained_path="pretrained_models/passt-s-f128-p16-s16-ap.468.pt", freqm=24, timem=96):
      
        
        super().__init__()
        
        # 构建模型结构但不加载预训练权重
        self.model = get_model_passt(
            "passt_s_p16_s16_128_ap468",
            input_tdim=1000,
            fstride=16,
            tstride=16,
            s_patchout_t=s_patchout_t,
            s_patchout_f=s_patchout_f,
            pretrained=False  # 关键：禁用自动下载
        )
        
        # 从指定路径加载预训练权重
        if pretrained_path is not None:
            if os.path.exists(pretrained_path):
                print(f"✓ 从指定路径加载预训练模型: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                
                # 处理可能的权重键名不匹配
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # 加载权重
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️ 缺失的键: {missing_keys}")
                if unexpected_keys:
                    print(f"⚠️ 意外的键: {unexpected_keys}")
                    
            else:
                raise FileNotFoundError(f"预训练模型文件不存在: {pretrained_path}")
        else:
            print("⚠️ 未提供预训练模型路径，模型将使用随机初始化权重")

        # freqm/timem: SpecAugment 参数，训练时增强鲁棒性 (eval 时自动禁用)
        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=32000,
            win_length=800,
            hopsize=320,
            n_fft=1024,
            freqm=freqm,
            timem=timem,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000
        )

    def forward(self, x):
        with torch.no_grad():
            mel = self.mel(x)

        tokens = self.model(mel[:, None])[-1] # get embedding, not token
        return tokens


class CutInputIntoSegmentsWrapper(nn.Module):
    def __init__(self, model, max_input_length, segment_length, hop_size):
        """
        Args:
            model (nn.Module): The PyTorch model to wrap.
            max_input_length (int): Maximum length of input the model can handle.
            segment_length (int): Length of each segment if input exceeds max_input_length.
            hop_size (int): Hop size for overlapping segmentation.
        """
        super().__init__()
        self.model = model
        self.max_input_length = max_input_length
        self.segment_length = segment_length
        self.hop_size = hop_size

    def forward(self, x):
        """Processes the input audio through the model, handling segmentation if needed."""
        batch_size, input_length = x.shape

        if input_length <= self.max_input_length:
            return self.model(x).unsqueeze(1)  # Add segment dimension

        # Split into overlapping segments
        segments = []
        indices = list(range(0, input_length - self.segment_length + 1, self.hop_size))
        for i in indices:
            segments.append(x[:, i:i + self.segment_length])

        segments = torch.stack(segments)  # Shape: (num_segments, batch_size, segment_length)
        outputs = self.model(segments.reshape(-1, self.segment_length))  # Process each segment
        outputs = outputs.view(len(indices), batch_size, -1).permute(1, 0, 2)   # Reshape back to (batch, num_segments , embedding_dim)

        # Return segments separately
        return outputs
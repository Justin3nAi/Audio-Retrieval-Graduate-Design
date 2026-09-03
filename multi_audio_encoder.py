"""
多音频编码器融合模块
同时使用PaSST、BEATs、CLAP提取音频特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAudioEncoderFusion(nn.Module):
    """
    多音频编码器融合模块
    支持三种融合策略：concat（拼接）、weighted（加权）、attention（注意力）
    """
    
    def __init__(
        self, 
        use_passt=True,
        use_beats=True, 
        use_clap=True,
        fusion_type='weighted',  # 'concat', 'weighted', 'attention'
        output_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        
        self.use_passt = use_passt
        self.use_beats = use_beats
        self.use_clap = use_clap
        self.fusion_type = fusion_type
        
        # 1. 初始化编码器
        self.encoders = nn.ModuleDict()
        self.encoder_dims = {}
        
        if use_passt:
            from d25_t6.passt import CutInputIntoSegmentsWrapper, PaSSTSNoOverlapWrapper
            self.encoders['passt'] = CutInputIntoSegmentsWrapper(
                PaSSTSNoOverlapWrapper(s_patchout_t=15, s_patchout_f=2),
                max_input_length=10*32000,
                segment_length=10*32000,
                hop_size=10*32000
            )
            self.encoder_dims['passt'] = 768
        
        if use_beats:
            # BEATs编码器（需要预先加载）
            # 这里假设已经加载好了BEATs模型
            self.encoders['beats'] = None  # 将在外部设置
            self.encoder_dims['beats'] = 768
            
            # 🔥 预先创建重采样器（避免每次forward都创建）
            import torchaudio
            self.beats_resampler = torchaudio.transforms.Resample(
                orig_freq=32000,
                new_freq=16000
            )
        
        if use_clap:
            # CLAP编码器（需要预先加载）
            self.encoders['clap'] = None  # 将在外部设置
            self.encoder_dims['clap'] = 512
        
        # 2. 融合策略
        if fusion_type == 'concat':
            # 拼接融合：直接拼接所有特征
            total_dim = sum(self.encoder_dims.values())
            self.fusion_proj = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim)
            )
            
        elif fusion_type == 'weighted':
            # 加权融合：学习每个编码器的权重
            num_encoders = len(self.encoder_dims)
            self.fusion_weights = nn.Parameter(torch.ones(num_encoders) / num_encoders)
            
            # 为每个编码器创建投影层（统一到相同维度）
            self.encoder_projs = nn.ModuleDict()
            for name, dim in self.encoder_dims.items():
                self.encoder_projs[name] = nn.Linear(dim, output_dim)
            
        elif fusion_type == 'attention':
            # 注意力融合：使用注意力机制动态融合
            # 先投影到相同维度
            self.encoder_projs = nn.ModuleDict()
            for name, dim in self.encoder_dims.items():
                self.encoder_projs[name] = nn.Linear(dim, output_dim)
            
            # 注意力层
            self.attention = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.Tanh(),
                nn.Linear(output_dim // 2, 1)
            )
        
        print(f"✅ MultiAudioEncoderFusion initialized")
        print(f"   Encoders: {list(self.encoder_dims.keys())}")
        print(f"   Fusion type: {fusion_type}")
        print(f"   Output dim: {output_dim}")
    
    def set_beats_encoder(self, beats_model):
        """设置BEATs编码器"""
        self.encoders['beats'] = beats_model
        print("✅ BEATs encoder set")
    
    def set_clap_encoder(self, clap_model):
        """设置CLAP编码器"""
        self.encoders['clap'] = clap_model
        print("✅ CLAP encoder set")
    
    def forward(self, audio, duration=None):
        """
        前向传播
        
        Args:
            audio: (batch, channels, samples) 音频输入
            duration: list of durations for each audio
        
        Returns:
            fused_features: (batch, output_dim) 融合后的特征
        """
        features = {}
        
        # 1. 提取每个编码器的特征
        if self.use_passt and self.encoders['passt'] is not None:
            with torch.amp.autocast('cuda', enabled=False):  # PaSST可能需要FP32
                passt_feat = self.encoders['passt'](audio.mean(1))  # (batch, segments, 768)
                
                # 确保聚合为2D张量
                if passt_feat.dim() == 3:
                    # 聚合多个片段
                    if duration is not None:
                        aggregated = []
                        for i, dur in enumerate(duration):
                            if dur <= 10:
                                aggregated.append(passt_feat[i, 0])
                            elif dur <= 20:
                                aggregated.append(passt_feat[i, :2].mean(0))
                            else:
                                aggregated.append(passt_feat[i].mean(0))
                        passt_feat = torch.stack(aggregated)  # (batch, 768)
                    else:
                        passt_feat = passt_feat.mean(1)  # 简单平均 (batch, 768)
                
                # 确保是2D张量
                if passt_feat.dim() != 2:
                    raise ValueError(f"PaSST特征维度错误: {passt_feat.shape}, 期望 (batch, 768)")
                
                features['passt'] = passt_feat
        
        if self.use_beats and self.encoders['beats'] is not None:
            # BEATs特征提取
            try:
                # 🔥 关键：在FP32精度下处理整个BEATs流程
                with torch.amp.autocast('cuda', enabled=False):
                    # 转为单声道
                    if audio.dim() == 3:
                        audio_mono = audio.mean(dim=1).float()  # (batch, samples) 强制FP32
                    else:
                        audio_mono = audio.float()
                    
                    # 归一化音频（逐样本归一化）
                    batch_size = audio_mono.shape[0]
                    audio_normalized = []
                    
                    for i in range(batch_size):
                        audio_sample = audio_mono[i]
                        audio_max = audio_sample.abs().max()
                        
                        if audio_max > 1e-8:
                            audio_sample = audio_sample / audio_max
                        else:
                            # 如果音频太小，使用零向量
                            audio_sample = torch.zeros_like(audio_sample)
                        
                        audio_normalized.append(audio_sample)
                    
                    audio_mono = torch.stack(audio_normalized)
                    
                    # 重采样：32kHz -> 16kHz
                    self.beats_resampler = self.beats_resampler.to(audio.device)
                    audio_16k = self.beats_resampler(audio_mono)
                    
                    # Padding/截断到固定长度
                    target_length = 160000  # 10秒 @ 16kHz
                    if audio_16k.shape[-1] < target_length:
                        pad_length = target_length - audio_16k.shape[-1]
                        audio_16k = F.pad(audio_16k, (0, pad_length), mode='constant', value=0)
                    elif audio_16k.shape[-1] > target_length:
                        audio_16k = audio_16k[..., :target_length]
                    
                    # Clamp到合理范围
                    audio_16k = torch.clamp(audio_16k, -1.0, 1.0)
                    
                    # 提取特征（在no_grad下）
                    with torch.no_grad():
                        beats_output = self.encoders['beats'].extract_features(audio_16k)
                    
                    # 处理输出
                    if isinstance(beats_output, tuple):
                        beats_feat = beats_output[0]
                    else:
                        beats_feat = beats_output
                    
                    # 聚合时间维度
                    if beats_feat.dim() == 3:
                        # (batch, time_steps, 768) -> (batch, 768)
                        beats_feat = beats_feat.mean(dim=1)
                    
                    # 最终检查
                    if torch.isnan(beats_feat).any() or torch.isinf(beats_feat).any():
                        print(f"⚠️ 警告: BEATs输出包含NaN/Inf，使用零向量")
                        beats_feat = torch.zeros(batch_size, 768, device=audio.device, dtype=torch.float32)
                    
                    features['beats'] = beats_feat
                
            except Exception as e:
                print(f"❌ BEATs特征提取失败: {e}")
                import traceback
                traceback.print_exc()
                # 使用零特征作为fallback
                features['beats'] = torch.zeros(audio.shape[0], 768, device=audio.device, dtype=torch.float32)
        
        if self.use_clap and self.encoders['clap'] is not None:
            # CLAP特征提取
            clap_feat = self.encoders['clap'](audio)  # (batch, 512)
            
            # 确保是2D张量
            if clap_feat.dim() != 2:
                raise ValueError(f"CLAP特征维度错误: {clap_feat.shape}, 期望 (batch, 512)")
            
            features['clap'] = clap_feat
        
        # 2. 融合特征
        # 调试：检查特征是否包含NaN
        for name, feat in features.items():
            if feat.dim() != 2:
                print(f"⚠️ 警告: {name}特征维度不正确: {feat.shape}")
            if torch.isnan(feat).any():
                print(f"❌ 错误: {name}特征包含NaN!")
                print(f"   特征统计: min={feat.min()}, max={feat.max()}, mean={feat.mean()}")
            if torch.isinf(feat).any():
                print(f"❌ 错误: {name}特征包含Inf!")
        
        if self.fusion_type == 'concat':
            # 拼接所有特征
            concat_feat = torch.cat([features[name] for name in sorted(features.keys())], dim=-1)
            fused = self.fusion_proj(concat_feat)
            
        elif self.fusion_type == 'weighted':
            # 加权融合
            projected_feats = []
            for i, name in enumerate(sorted(features.keys())):
                proj_feat = self.encoder_projs[name](features[name])
                weighted_feat = proj_feat * torch.sigmoid(self.fusion_weights[i])
                projected_feats.append(weighted_feat)
            
            fused = sum(projected_feats)
            
        elif self.fusion_type == 'attention':
            # 注意力融合
            projected_feats = []
            for name in sorted(features.keys()):
                proj_feat = self.encoder_projs[name](features[name])
                # 检查投影后的特征
                if torch.isnan(proj_feat).any():
                    print(f"❌ 错误: {name}投影后包含NaN!")
                projected_feats.append(proj_feat)
            
            # Stack: (batch, num_encoders, output_dim)
            stacked_feats = torch.stack(projected_feats, dim=1)
            
            # 计算注意力权重
            attn_scores = self.attention(stacked_feats)  # (batch, num_encoders, 1)
            
            # 检查注意力分数
            if torch.isnan(attn_scores).any():
                print(f"❌ 错误: 注意力分数包含NaN!")
                print(f"   stacked_feats统计: min={stacked_feats.min()}, max={stacked_feats.max()}")
            
            attn_weights = F.softmax(attn_scores, dim=1)
            
            # 检查注意力权重
            if torch.isnan(attn_weights).any():
                print(f"❌ 错误: 注意力权重包含NaN!")
                print(f"   attn_scores统计: min={attn_scores.min()}, max={attn_scores.max()}")
            
            # 加权求和
            fused = (stacked_feats * attn_weights).sum(dim=1)  # (batch, output_dim)
        
        # 3. L2归一化
        fused = F.normalize(fused, p=2, dim=-1)
        
        # 调试：检查融合后的特征
        if torch.isnan(fused).any():
            print(f"❌ 错误: 融合后的特征包含NaN!")
            print(f"   融合类型: {self.fusion_type}")
            print(f"   特征统计: min={fused.min()}, max={fused.max()}")
        
        return fused


def load_beats_model(model_path, device='cuda'):
    """
    加载BEATs预训练模型
    
    Args:
        model_path: BEATs模型路径
        device: 设备
    
    Returns:
        beats_model: BEATs模型包装器
    """
    try:
        # 尝试导入BEATs
        import sys
        import os
        
        # 添加BEATs库路径
        beats_lib_path = '/root/autodl-tmp/ProjectAR/beats_lib'
        if beats_lib_path not in sys.path:
            sys.path.insert(0, beats_lib_path)
        
        print(f"📥 BEATs库路径: {beats_lib_path}")
        print(f"   文件存在: {os.path.exists(os.path.join(beats_lib_path, 'BEATs.py'))}")
        
        # 检查依赖
        try:
            import fairseq
            print(f"✅ fairseq已安装: {fairseq.__version__}")
        except ImportError:
            print("⚠️ fairseq未安装，BEATs可能需要此依赖")
            print("   请运行: pip install fairseq")
        
        try:
            # 尝试从本地beats_lib导入
            print("📥 尝试导入BEATs...")
            from BEATs import BEATs, BEATsConfig
            print(f"✅ 从本地导入BEATs成功: {beats_lib_path}")
        except ImportError as e:
            print(f"❌ 从本地导入BEATs失败: {e}")
            # 尝试从安装的beats包导入
            try:
                from beats import BEATs, BEATsConfig
                print("✅ 从已安装的beats包导入")
            except ImportError as e2:
                print(f"❌ 从已安装包导入也失败: {e2}")
                print("\n详细错误信息:")
                import traceback
                traceback.print_exc()
                print("\n请检查:")
                print(f"   1. {beats_lib_path}/BEATs.py 是否存在")
                print("   2. 是否安装了 fairseq: pip install fairseq")
                print("   3. 或使用 --no-use_beats 禁用BEATs")
                return None
        
        # 加载模型
        print(f"📥 加载BEATs模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        cfg = BEATsConfig(checkpoint['cfg'])
        beats_model = BEATs(cfg)
        beats_model.load_state_dict(checkpoint['model'])
        beats_model.eval()
        beats_model.to(device)
        
        # 冻结参数
        for param in beats_model.parameters():
            param.requires_grad = False
        
        print(f"✅ BEATs model loaded from {model_path}")
        return beats_model
        
    except Exception as e:
        print(f"❌ Failed to load BEATs: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_clap_model(model_name='laion/clap-htsat-unfused', device='cuda'):
    """
    加载CLAP预训练模型
    
    Args:
        model_name: CLAP模型名称或路径（支持HuggingFace模型名或本地路径）
        device: 设备
    
    Returns:
        clap_wrapper: CLAP包装器（包含模型和处理器）
    """
    try:
        from transformers import ClapModel, ClapProcessor
        import torchaudio
        
        # 加载模型和处理器
        print(f"📥 Loading CLAP from: {model_name}")
        clap_model = ClapModel.from_pretrained(model_name)
        clap_processor = ClapProcessor.from_pretrained(model_name)
        
        clap_model.eval()
        clap_model.to(device)
        
        # 冻结参数
        for param in clap_model.parameters():
            param.requires_grad = False
        
        # 创建包装器
        class CLAPWrapper(nn.Module):
            def __init__(self, model, processor, device):
                super().__init__()
                self.model = model
                self.processor = processor
                self.device = device
                self.target_sr = 48000  # CLAP使用48kHz
            
            def forward(self, audio_tensor):
                """
                Args:
                    audio_tensor: (batch, channels, samples) 或 (batch, samples)
                
                Returns:
                    audio_embeds: (batch, 512) CLAP音频特征
                """
                # 1. 处理输入格式
                if audio_tensor.dim() == 3:
                    # (batch, channels, samples) -> (batch, samples) 转为单声道
                    audio_tensor = audio_tensor.mean(dim=1)
                
                batch_size = audio_tensor.shape[0]
                
                # 2. 重采样到48kHz（如果需要）
                # 假设输入是32kHz
                if audio_tensor.shape[-1] > 0:
                    # 使用torchaudio重采样
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=32000, 
                        new_freq=self.target_sr
                    ).to(self.device)
                    audio_tensor = resampler(audio_tensor)
                
                # 3. 转换为numpy并处理
                audio_np = audio_tensor.cpu().numpy()
                
                # 4. 使用processor处理音频
                inputs = self.processor(
                    audios=list(audio_np),
                    return_tensors="pt",
                    sampling_rate=self.target_sr,
                    padding=True
                )
                
                # 5. 移到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 6. 提取音频特征
                with torch.no_grad():
                    audio_embeds = self.model.get_audio_features(**inputs)
                
                return audio_embeds  # (batch, 512)
        
        clap_wrapper = CLAPWrapper(clap_model, clap_processor, device)
        
        print(f"✅ CLAP model loaded: {model_name}")
        print(f"   Audio embedding dim: 512")
        print(f"   Target sample rate: 48000 Hz")
        
        return clap_wrapper
        
    except Exception as e:
        print(f"❌ Failed to load CLAP: {e}")
        import traceback
        traceback.print_exc()
        return None


# 使用示例
if __name__ == '__main__':
    # 创建融合模块
    fusion_module = MultiAudioEncoderFusion(
        use_passt=True,
        use_beats=True,
        use_clap=True,
        fusion_type='attention',  # 使用注意力融合
        output_dim=1024
    )
    
    # 加载BEATs和CLAP
    beats_model = load_beats_model('/path/to/BEATs_iter3_plus_AS2M.pt')
    clap_model = load_clap_model('laion/clap-htsat-unfused')
    
    if beats_model is not None:
        fusion_module.set_beats_encoder(beats_model)
    
    if clap_model is not None:
        fusion_module.set_clap_encoder(clap_model)
    
    # 测试
    dummy_audio = torch.randn(4, 2, 320000)  # (batch=4, channels=2, samples)
    dummy_duration = [8.5, 12.3, 15.7, 9.2]
    
    with torch.no_grad():
        fused_features = fusion_module(dummy_audio, dummy_duration)
    
    print(f"Fused features shape: {fused_features.shape}")  # (4, 1024)


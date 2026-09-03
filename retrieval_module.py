import copy
import math
import string
from typing import Any
import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from transformers import RobertaTokenizer, RobertaModel
from d25_t6.passt import CutInputIntoSegmentsWrapper, PaSSTSNoOverlapWrapper
from d25_t6.moe_module import SimpleMoEFFN

class CrossModalAttention(torch.nn.Module):
    """双模态交叉注意力机制，增强音频-文本特征交互"""

    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1, use_moe=False, num_experts=4, top_k=2):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        # 🔥 MoE FFN 或标准 FFN
        self.use_moe = use_moe
        if use_moe:
            self.ffn = SimpleMoEFFN(embed_dim, num_experts=num_experts, top_k=top_k, dropout=dropout)
        else:
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim * 4),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(embed_dim * 4, embed_dim),
                torch.nn.Dropout(dropout)
            )
    def forward(self, query, key_value):
        """
        query: (batch, embed_dim) - 查询模态
        key_value: (batch, embed_dim) - 键值模态
        """
        # 添加序列维度
        query = query.unsqueeze(1)  # (batch, 1, embed_dim)
        key_value = key_value.unsqueeze(1)  # (batch, 1, embed_dim)
        # 交叉注意力
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        query = self.norm1(query + attn_output)
        # 前馈网络（MoE或标准FFN）
        query_2d = query.squeeze(1)
        if self.use_moe:
            ffn_output = self.ffn(query_2d)
            ffn_output = ffn_output.unsqueeze(1)
        else:
            ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        return output.squeeze(1)  # (batch, embed_dim)

class ImprovedProjectionHead(torch.nn.Module):
    """简化的投影头，2层MLP更容易训练"""

    def __init__(self, input_dim, output_dim=1024, hidden_dim=1024, dropout=0.1):
        super().__init__()
        # 简化为2层，hidden_dim也减小
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class AttentiveAudioAggregation(torch.nn.Module):
    """注意力加权的音频特征聚合 - 替代简单平均"""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(embed_dim // 2, 1)
        )
    def forward(self, audio_embeddings, num_segments):
        """
        audio_embeddings: (batch, max_segments, embed_dim)
        num_segments: list of actual segment counts per sample
        """
        batch_size = audio_embeddings.size(0)
        max_segments = audio_embeddings.size(1)
        # 创建mask
        mask = torch.zeros(batch_size, max_segments, device=audio_embeddings.device, dtype=torch.bool)
        for i, n in enumerate(num_segments):
            if n < max_segments:
                mask[i, n:] = True
        # 🔥 修复：使用float32避免overflow
        attn_weights = self.attention(audio_embeddings)  # (batch, max_segments, 1)
        attn_weights = attn_weights.float()
        # 使用-1e4而不是-1e9，对float16安全
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e4)
        attn_weights = F.softmax(attn_weights, dim=1)
        # 转回原始dtype
        attn_weights = attn_weights.to(audio_embeddings.dtype)
        # 加权聚合
        aggregated = (audio_embeddings * attn_weights).sum(dim=1)
        return aggregated

class AttentionPooling(torch.nn.Module):
    """注意力池化 - 替代单一[CLS] token"""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.attention = torch.nn.Linear(hidden_size, 1)
    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """
        # 计算注意力分数
        attn_scores = self.attention(hidden_states)  # (batch, seq_len, 1)
        # 🔥 修复：使用float32避免overflow，-1e4对float16安全
        attn_scores = attn_scores.float()
        # Mask padding tokens (使用-1e4而不是-1e9，对float16安全)
        attn_scores = attn_scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0, -1e4
        )
        attn_weights = F.softmax(attn_scores, dim=1)
        # 转回原始dtype
        attn_weights = attn_weights.to(hidden_states.dtype)
        # 加权池化
        pooled = (hidden_states * attn_weights).sum(dim=1)
        return pooled

class AudioRetrievalModel(pl.LightningModule):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(kwargs)
        # 🔥 多音频编码器融合（新增）
        self.use_multi_encoder = kwargs.get('use_multi_encoder', False)
        if self.use_multi_encoder:
            # 使用多编码器融合
            from d25_t6.multi_audio_encoder import MultiAudioEncoderFusion
            self.audio_embedding_model = MultiAudioEncoderFusion(
                use_passt=kwargs.get('use_passt', True),
                use_beats=kwargs.get('use_beats', True),
                use_clap=kwargs.get('use_clap', True),
                fusion_type=kwargs.get('fusion_type', 'attention'),  # 'concat', 'weighted', 'attention'
                output_dim=1024,
                dropout=kwargs.get('dropout_rate', 0.1)
            )
            # 多编码器直接输出1024维，不需要额外投影
            self.audio_projection = torch.nn.Identity()
            print(f"✅ Multi-encoder fusion enabled")
            print(f"   Fusion type: {kwargs.get('fusion_type', 'attention')}")
        else:
            # 使用单一编码器（PaSST）
            self.audio_embedding_model = CutInputIntoSegmentsWrapper(
                PaSSTSNoOverlapWrapper(
                    s_patchout_t=kwargs['s_patchout_t'],
                    s_patchout_f=kwargs['s_patchout_f']
                ),
                max_input_length=10*32000,
                segment_length=10*32000,
                hop_size=10*32000
            )
        # 投影层 - 使用改进的投影头，增加dropout
        use_mlp = kwargs.get('use_mlp_projection', False)
        use_improved_projection = kwargs.get('use_improved_projection', True)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        if use_improved_projection:
            self.audio_projection = ImprovedProjectionHead(768, 1024, hidden_dim=1024, dropout=dropout_rate)
        elif use_mlp:
            self.audio_projection = torch.nn.Sequential(
                torch.nn.Linear(768, 1024),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(1024, 1024)
            )
        else:
            self.audio_projection = torch.nn.Linear(768, 1024)
        # text encoder
        model_name = 'roberta-base' if kwargs['roberta_base'] else 'roberta-large'
        local_model_path = os.path.join('/root/autodl-tmp/huggingface_cache', model_name)
        # 检查本地模型是否存在，如果不存在则回退到在线加载
        if not os.path.exists(local_model_path):
            print(f"⚠️ 警告: 本地模型路径不存在 {local_model_path}, 回退到在线加载")
            local_model_path = model_name  # 回退到在线加载
        else:
            print(f"✅ 从本地路径加载模型: {local_model_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(local_model_path)
        # 增加dropout防止过拟合
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.text_embedding_model = RobertaModel.from_pretrained(
            local_model_path,
            add_pooling_layer=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            output_hidden_states=True  # 启用多层特征提取
        )
        text_dim = 768 if kwargs['roberta_base'] else 1024
        if use_improved_projection:
            self.text_projection = ImprovedProjectionHead(text_dim, 1024, hidden_dim=1024, dropout=dropout_rate)
        elif use_mlp:
            self.text_projection = torch.nn.Sequential(
                torch.nn.Linear(text_dim, 1024),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(1024, 1024)
            )
        else:
            self.text_projection = torch.nn.Linear(text_dim, 1024)
        # 🔥 新增：注意力聚合模块
        self.use_attentive_aggregation = kwargs.get('use_attentive_aggregation', True)
        if self.use_attentive_aggregation:
            self.audio_aggregation = AttentiveAudioAggregation(embed_dim=768)
        # 🔥 新增：注意力池化模块
        self.use_attention_pooling = kwargs.get('use_attention_pooling', True)
        if self.use_attention_pooling:
            self.text_pooling = AttentionPooling(hidden_size=text_dim)
        # 交叉注意力机制（可选）
        self.use_cross_attention = kwargs.get('use_cross_attention', True)
        use_moe = kwargs.get('use_moe', False)
        num_experts = kwargs.get('num_experts', 4)
        top_k = kwargs.get('top_k', 2)
        if self.use_cross_attention:
            self.audio_to_text_attn = CrossModalAttention(
                1024, num_heads=8, dropout=dropout_rate, 
                use_moe=use_moe, num_experts=num_experts, top_k=top_k
            )
            self.text_to_audio_attn = CrossModalAttention(
                1024, num_heads=8, dropout=dropout_rate,
                use_moe=use_moe, num_experts=num_experts, top_k=top_k
            )
        # temperature parameter
        initial_tau = torch.zeros((1,)) + kwargs['initial_tau']
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=kwargs['tau_trainable'])
        # EMA (Exponential Moving Average) for better generalization
        self.use_ema = kwargs.get('use_ema', True)
        if self.use_ema:
            self.ema_decay = kwargs.get('ema_decay', 0.999)
            self.ema_model = None  # 将在第一次训练步骤中初始化
        # 🔥 在线知识蒸馏 (Online Distillation with EMA Teacher)
        self.use_online_distillation = kwargs.get('use_online_distillation', True)
        if self.use_online_distillation:
            from d25_t6.online_distillation import OnlineDistillationLoss, EMATeacher
            self.online_distill_loss_fn = OnlineDistillationLoss(
                temperature=kwargs.get('distill_temperature', 2.0),
                alpha=kwargs.get('distill_alpha', 1.0)
            )
            self.ema_teacher = None  # 将在第一次训练步骤中初始化
            print(f"✅ Online distillation enabled (EMA-based)")
        # 🔥 聚类引导分类 (Clustering-Guided Classification)
        self.use_clustering_classification = kwargs.get('use_clustering_classification', True)
        if self.use_clustering_classification:
            from d25_t6.clustering_classification import ClusteringClassifier
            num_clusters = kwargs.get('num_clusters', 50)
            self.clustering_classifier = ClusteringClassifier(
                embed_dim=1024,
                num_clusters=num_clusters,
                dropout=dropout_rate
            )
            self.clustering_weight = kwargs.get('clustering_weight', 0.05)
            self.clusterer = None  # 将在train.py中初始化
            print(f"✅ Clustering classification enabled ({num_clusters} clusters)")
        # 🔥 多粒度语义对齐 (Multi-Granularity Semantic Alignment)
        self.use_multi_granularity = kwargs.get('use_multi_granularity', False)
        if self.use_multi_granularity:
            from d25_t6.multi_granularity_alignment import MultiGranularityAlignment
            self.multi_granularity_alignment = MultiGranularityAlignment(
                embed_dim=1024,
                num_heads=8,
                dropout=dropout_rate,
                global_weight=kwargs.get('mg_global_weight', 0.7),
                local_weight=kwargs.get('mg_local_weight', 0.3)
            )
            print(f"✅ Multi-Granularity Alignment enabled")
            print(f"   Global weight: {kwargs.get('mg_global_weight', 0.7)}")
            print(f"   Local weight: {kwargs.get('mg_local_weight', 0.3)}")
        # 🔥 AudioCLIP教师蒸馏 (方案C - 最强优化)
        self.use_audioclip_distillation = kwargs.get('use_audioclip_distillation', False)
        self.audioclip_teacher = None  # 将在train.py中初始化
        self.audioclip_distillation_loss_fn = None  # 将在train.py中初始化
        # 🔥 多教师蒸馏 (Multi-Teacher Distillation) - 保留但默认禁用
        self.use_multi_teacher_distillation = kwargs.get('use_multi_teacher_distillation', False)
        self.teacher_ensemble = None  # 将在train.py中初始化
        self.multi_teacher_distillation_loss_fn = None  # 将在train.py中初始化
        # Ensemble distillation initialization
        self.use_ensemble_distillation = kwargs.get('use_ensemble_distillation', False)
        self.teacher_models = []
        self._teachers_loaded = False  # lazy load flag
        self.ensemble_distill_loss_fn = None
        if self.use_ensemble_distillation:
            # 🔥 优先使用预训练教师模型
            use_pretrained_teachers = kwargs.get('use_pretrained_teachers', False)
            if use_pretrained_teachers:
                print("=" * 60)
                print("Loading PRETRAINED teacher models for ensemble distillation...")
                print("=" * 60)
                # 预训练教师配置
                # 只用成功加载的 CLAP 教师（2个）
                # 3个教师：2个CLAP + 1个BEATs
                # 只用 2 个 CLAP 教师（移除 BEATs，它没有真正的文本能力）
                pretrained_configs = [
                    {'type': 'clap', 'path': '/root/autodl-tmp/teacher_models/clap/clap-htsat-unfused'},
                    {'type': 'clap', 'path': '/root/autodl-tmp/teacher_models/clap/clap-larger'},
                ]
                from d25_t6.pretrained_teacher_loader import load_pretrained_teachers
                from d25_t6.knowledge_distillation import EnsembleDistillationLoss
                self.teacher_models = load_pretrained_teachers(pretrained_configs)
                if len(self.teacher_models) > 0:
                    self.ensemble_distill_loss_fn = EnsembleDistillationLoss(
                        temperature=kwargs.get('ensemble_distill_temperature', 1.0)
                    )
                    self.ensemble_distill_weight = kwargs.get('ensemble_distill_weight', 0.1)
                    print("=" * 60)
                    print(f"Pretrained ensemble distillation enabled")
                    print(f"   Teachers: {len(self.teacher_models)}")
                    print(f"   Weight: {self.ensemble_distill_weight}")
                    print(f"   Temperature: {kwargs.get('ensemble_distill_temperature', 2.0)}")
                    print("=" * 60)
                else:
                    print("WARNING: No pretrained teachers loaded, distillation disabled")
                    self.use_ensemble_distillation = False
            else:
                # 使用自己训练的教师模型
                teacher_checkpoints = kwargs.get('teacher_checkpoints', [])
                if teacher_checkpoints:
                    print("=" * 60)
                    print(f"Loading {len(teacher_checkpoints)} trained teacher models...")
                    print("=" * 60)
                    from d25_t6.knowledge_distillation import EnsembleDistillationLoss
                    for i, ckpt_path in enumerate(teacher_checkpoints):
                        print(f"  Teacher {i+1}: {ckpt_path}")
                        try:
                            teacher = AudioRetrievalModel.load_from_checkpoint(
                                ckpt_path, map_location='cpu', strict=False
                            )
                            teacher.eval()
                            for param in teacher.parameters():
                                param.requires_grad = False
                            self.teacher_models.append(teacher)
                            print(f"     SUCCESS")
                        except Exception as e:
                            print(f"     FAILED: {e}")
                    if len(self.teacher_models) > 0:
                        self.ensemble_distill_loss_fn = EnsembleDistillationLoss(
                            temperature=kwargs.get('ensemble_distill_temperature', 1.0)
                        )
                        self.ensemble_distill_weight = kwargs.get('ensemble_distill_weight', 1.0)
                        print("=" * 60)
                        print(f"Ensemble distillation enabled")
                        print(f"   Teachers: {len(self.teacher_models)}")
                        print(f"   Weight: {self.ensemble_distill_weight}")
                        print(f"   Temperature: {kwargs.get('ensemble_distill_temperature', 1.0)}")
                        print("=" * 60)
                    else:
                        print("WARNING: No teachers loaded, distillation disabled")
                        self.use_ensemble_distillation = False
                else:
                    print("WARNING: No teacher checkpoints provided")
                    self.use_ensemble_distillation = False
        # 🔥 MoE负载均衡损失权重
        self.moe_load_balance_weight = kwargs.get('moe_load_balance_weight', 0.01)
        self.validation_outputs = []
        # 🔥 数据增强（SpecAugment + Mixup）
        self.use_augmentation = kwargs.get('use_augmentation', False)
        if self.use_augmentation:
            from d25_t6.audio_augmentation import AudioAugmentation
            self.augmentation = AudioAugmentation(
                use_spec_augment=True,
                use_mixup=False,  # Mixup breaks contrastive learning
                use_time_stretch=False,  # 太慢，不用
                use_noise=False,
                spec_augment={'p': 0.5, 'freq_mask_param': 20, 'time_mask_param': 30},
                mixup={'alpha': 0.2, 'p': 0.3}
            )
        # 🔥 改进的损失函数
        self.use_improved_loss = kwargs.get('use_improved_loss', True)
        if self.use_improved_loss:
            from d25_t6.improved_losses import CombinedLoss
            self.improved_loss_fn = CombinedLoss(
                use_focal=True,
                use_hard_negative=True,
                use_infonce=False,  # 已经有基础 InfoNCE
                focal_weight=0.6,
                hard_neg_weight=0.4,
                focal={'temperature': 0.07, 'gamma': 2.0, 'alpha': 0.25},
                hard_neg={'temperature': 0.07, 'hard_ratio': 0.5}
            )
        self.kwargs = kwargs
        # 损失函数配置
        self.loss_type = kwargs.get('loss_type', 'improved_infonce')  # 'infonce', 'improved_infonce', 'focal'
        self.hard_negative_weight = kwargs.get('hard_negative_weight', 0.5)
        self.focal_gamma = kwargs.get('focal_gamma', 2.0)
        self.compile_model()
    def compile_model(self):
        """Apply torch.compile() if GPU is recent"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()  # Get current GPU device
            properties = torch.cuda.get_device_properties(device)
            if properties.major >= 7 and self.kwargs['compile'] == True:
                print("Compiling Models")
                self.text_embedding_model = torch.compile(self.text_embedding_model)
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)
    def forward(self, batch, **kwargs) -> Any:
        """接受 **kwargs 以兼容可能传入的额外参数（如 use_cross_attention）"""
        # embed audio & text
        text_embeddings = self.forward_text(batch)
        audio_embeddings = self.forward_audio(batch)
        # 交叉注意力机制（暂时禁用，因为会导致训练不稳定）
        # 如果需要启用，请确保模型已经充分预训练
        if self.use_cross_attention and self.training:
            current_epoch = self.current_epoch
            cross_attn_warmup_epochs = self.kwargs.get('cross_attn_warmup_epochs', 100)  # 设置很大，实际禁用
            if current_epoch >= cross_attn_warmup_epochs:
                # 使用新的autocast API
                with torch.amp.autocast('cuda', enabled=False):
                    audio_embeddings_fp32 = audio_embeddings.float()
                    text_embeddings_fp32 = text_embeddings.float()
                    audio_embeddings_enhanced = self.audio_to_text_attn(audio_embeddings_fp32, text_embeddings_fp32)
                    text_embeddings_enhanced = self.text_to_audio_attn(text_embeddings_fp32, audio_embeddings_fp32)
                    # 使用非常小的权重融合
                    audio_embeddings = F.normalize(audio_embeddings_fp32 + 0.05 * audio_embeddings_enhanced, p=2, dim=-1)
                    text_embeddings = F.normalize(text_embeddings_fp32 + 0.05 * text_embeddings_enhanced, p=2, dim=-1)
        return audio_embeddings, text_embeddings
    def forward_audio(self, batch, return_frames=False):
        # Apply data augmentation during training
        if self.training and hasattr(self, 'augmentation') and self.use_augmentation:
            batch['audio'] = self.augmentation(batch['audio'])
        # 🔥 多编码器融合
        if self.use_multi_encoder:
            # 多编码器已经处理了聚合和投影
            audio_embeddings = self.audio_embedding_model(
                batch['audio'], 
                duration=batch.get('duration')
            )
            # 已经归一化，直接返回
            return audio_embeddings
        # 单编码器（原始逻辑）
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1)) # forward
        # 🔥 改进：使用注意力加权聚合，而不是简单平均
        if self.use_attentive_aggregation and audio_embeddings.dim() == 3:
            # audio_embeddings: (batch, num_segments, embed_dim)
            # 计算每个样本的实际segment数量
            num_segments = []
            for duration in batch['duration']:
                if duration <= 10:
                    num_segments.append(1)
                elif duration <= 20:
                    num_segments.append(2)
                else:
                    num_segments.append(audio_embeddings.size(1))
            # 使用注意力聚合
            audio_embeddings = self.audio_aggregation(audio_embeddings, num_segments)
        else:
            # 回退到原始的简单聚合方式
            aggregated = []
            for i, duration in enumerate(batch['duration']):
                if duration <= 10:
                    aggregated.append(audio_embeddings[i, 0])
                elif duration <= 20:
                    aggregated.append(audio_embeddings[i, :2].mean(-2))
                else:
                    aggregated.append(audio_embeddings[i].mean(-2))
            audio_embeddings = torch.stack(aggregated)
        audio_embeddings = self.audio_projection(audio_embeddings) # project to same dimension
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=-1) # normalize
        # 🔥 多粒度对齐：支持返回帧级特征
        if return_frames:
            return audio_embeddings, None
        return audio_embeddings
    def forward_text(self, batch, return_tokens=False):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))
        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )
        outputs = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )
        # 🔥 改进：使用注意力池化，而不是只用[CLS] token
        if self.use_attention_pooling:
            # 使用注意力池化所有token
            token_embeddings = outputs[0]  # (batch, seq_len, hidden_size)
            sentence_features = self.text_pooling(token_embeddings, tokenized['attention_mask'].to(device))
        else:
            # 多层特征融合策略
            use_multi_layer = self.kwargs.get('use_multi_layer_text', True)
            if use_multi_layer and outputs.hidden_states is not None:
                # 使用最后4层的加权平均
                hidden_states = outputs.hidden_states
                # 取最后4层: -4, -3, -2, -1
                last_4_layers = torch.stack(hidden_states[-4:], dim=0)  # (4, batch, seq, dim)
                # 加权平均（后面的层权重更大）
                weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device).view(4, 1, 1, 1)
                weighted_layers = (last_4_layers * weights).sum(dim=0)  # (batch, seq, dim)
                sentence_features = weighted_layers[:, 0, :]  # 使用 [CLS] token
            else:
                # 使用 [CLS] token (RoBERTa 预训练的标准句子表示方式)
                token_embeddings = outputs[0]
                sentence_features = token_embeddings[:, 0, :]
        # project
        sentence_features = self.text_projection(sentence_features)
        # normalize
        sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=-1)
        # 🔥 多粒度对齐：支持返回词级特征
        if return_tokens:
            text_tokens = outputs[0]  # (batch, seq_len, hidden_size)
            text_mask = (tokenized['attention_mask'] == 0).to(device)
            # 投影词级特征到1024维
            batch_size, seq_len, hidden_size = text_tokens.shape
            text_tokens_flat = text_tokens.view(-1, hidden_size)
            text_tokens_proj = self.text_projection(text_tokens_flat)
            text_tokens_proj = text_tokens_proj.view(batch_size, seq_len, -1)
            text_tokens_proj = torch.nn.functional.normalize(text_tokens_proj, p=2, dim=-1)
            return sentence_features, text_tokens_proj, text_mask
        return sentence_features
    def training_step(self, batch, batch_idx):
        self.lr_scheduler_step(batch_idx)
        # 🔥 初始化 EMA Teacher（在第一个训练步骤）
        if self.use_online_distillation and self.ema_teacher is None:
            from d25_t6.online_distillation import EMATeacher
            self.ema_teacher = EMATeacher(self, decay=self.ema_decay)
            print("✅ EMA Teacher initialized for online distillation")
        # 初始化 EMA（在第一个训练步骤）
        if self.use_ema and self.ema_model is None:
            self.ema_model = copy.deepcopy(self.state_dict())
        # 学生模型前向传播
        # 🔥 多粒度对齐训练
        if self.use_multi_granularity:
            # 提取全局和局部特征
            audio_result = self.forward_audio(batch, return_frames=True)
            text_result = self.forward_text(batch, return_tokens=True)
            # 检查是否成功提取局部特征
            if isinstance(text_result, tuple) and len(text_result) == 3:
                audio_global = audio_result[0]
                text_global, text_tokens, text_mask = text_result
                if text_tokens is not None and len(audio_global) > 1:
                    # 多粒度对齐
                    combined_sim, global_sim, local_sim = self.multi_granularity_alignment(
                        audio_global,
                        text_global,
                        None,  # audio_frames (暂不使用)
                        text_tokens,
                        audio_mask=None,
                        text_mask=text_mask,
                        return_components=True
                    )
                    # 计算多粒度损失
                    from d25_t6.multi_granularity_alignment import MultiGranularityRetrievalLoss
                    # 构建标签
                    paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) 
                                     for i, p in enumerate(batch['fname'])])
                    labels = torch.tensor(paths[None, :] == paths[:, None], device=combined_sim.device)
                    loss_fn = MultiGranularityRetrievalLoss(
                        temperature=torch.abs(self.tau).item(),
                        global_weight=self.kwargs.get('mg_global_weight', 0.7),
                        local_weight=self.kwargs.get('mg_local_weight', 0.3)
                    )
                    total_loss, loss_dict = loss_fn(global_sim, local_sim, labels)
                    # 🔥 检查损失是否有效
                    if total_loss.item() > 0:
                        # 记录损失
                        self.log("train/loss", total_loss, batch_size=len(audio_global), sync_dist=True, prog_bar=True)
                        self.log("train/global_loss", loss_dict['global_loss'], batch_size=len(audio_global), sync_dist=True)
                        if 'local_loss' in loss_dict:
                            self.log("train/local_loss", loss_dict['local_loss'], batch_size=len(audio_global), sync_dist=True)
                        # 用于其他损失计算
                        audio_embeddings = audio_global
                        text_embeddings = text_global
                        student_similarity = torch.matmul(audio_embeddings, text_embeddings.T)
                        contrastive_loss = total_loss
                    else:
                        # 损失为0，回退到标准训练
                        audio_embeddings, text_embeddings = self.forward(batch)
                        student_similarity = torch.matmul(audio_embeddings, text_embeddings.T)
                        contrastive_loss = self.compute_loss(audio_embeddings, text_embeddings, batch)
                        total_loss = contrastive_loss
                        self.log("train/loss", total_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
                else:
                    # 回退到标准训练
                    audio_embeddings, text_embeddings = self.forward(batch)
                    student_similarity = torch.matmul(audio_embeddings, text_embeddings.T)
                    contrastive_loss = self.compute_loss(audio_embeddings, text_embeddings, batch)
                    total_loss = contrastive_loss
                    self.log("train/loss", total_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
            else:
                # 回退到标准训练
                audio_embeddings, text_embeddings = self.forward(batch)
                student_similarity = torch.matmul(audio_embeddings, text_embeddings.T)
                contrastive_loss = self.compute_loss(audio_embeddings, text_embeddings, batch)
                total_loss = contrastive_loss
                self.log("train/loss", total_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        else:
            # 标准训练流程
            audio_embeddings, text_embeddings = self.forward(batch)
            student_similarity = torch.matmul(audio_embeddings, text_embeddings.T)
            contrastive_loss = self.compute_loss(audio_embeddings, text_embeddings, batch)
            total_loss = contrastive_loss
            self.log("train/loss", total_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        # 🔥 集成知识蒸馏损失（Kim论文）
        # === Load trained checkpoint teachers lazily ===
        if self.use_ensemble_distillation and not self._teachers_loaded:
            self._teachers_loaded = True
            trained_ckpts = self.hparams.get('teacher_checkpoints', [])
            if trained_ckpts and len(self.teacher_models) == 0:
                print("=" * 60)
                print(f"Loading {len(trained_ckpts)} trained teacher models...")
                print("=" * 60)
                from d25_t6.knowledge_distillation import EnsembleDistillationLoss
                for ckpt_path in trained_ckpts:
                    print(f"  Loading: {ckpt_path}")
                    try:
                        teacher = AudioRetrievalModel.load_from_checkpoint(
                            ckpt_path,
                            map_location=self.device,
                            strict=False
                        )
                        teacher.eval()
                        teacher._teachers_loaded = True  # prevent recursive init
                        for p in teacher.parameters():
                            p.requires_grad = False
                        self.teacher_models.append(teacher)
                        print(f"     SUCCESS")
                    except Exception as e:
                        print(f"     FAILED: {e}")
                if len(self.teacher_models) > 0 and self.ensemble_distill_loss_fn is None:
                    self.ensemble_distill_loss_fn = EnsembleDistillationLoss(
                        temperature=self.hparams.get('ensemble_distill_temperature', 1.0)
                    )
                    self.ensemble_distill_weight = self.hparams.get('ensemble_distill_weight', 1.0)
                    print("=" * 60)
                    print(f"Ensemble distillation enabled: {len(self.teacher_models)} trained teachers")
                    print(f"   Weight: {self.ensemble_distill_weight}")
                    print("=" * 60)
        if self.use_ensemble_distillation and len(self.teacher_models) > 0:
            teacher_similarities = []
            device = audio_embeddings.device
            with torch.no_grad():
                for idx, teacher in enumerate(self.teacher_models):
                    # 确保教师在正确的设备上
                    if next(teacher.parameters()).device != device:
                        teacher = teacher.to(device)
                    # 教师前向传播
                    try:
                        import time
                        start_time = time.time()
                        teacher_audio_emb, teacher_text_emb = teacher.forward(batch)
                        elapsed = time.time() - start_time
                        print(f"[Teacher {idx+1}] Forward: {elapsed:.2f}s, Audio: {teacher_audio_emb.shape}, Text: {teacher_text_emb.shape}")
                        teacher_sim = torch.matmul(teacher_audio_emb, teacher_text_emb.T)
                        teacher_similarities.append(teacher_sim)
                    except Exception as e:
                        print(f"⚠️  教师模型前向传播失败: {e}")
            # 如果成功获取教师输出，计算蒸馏损失
            if len(teacher_similarities) > 0:
                ensemble_distill_loss = self.ensemble_distill_loss_fn(student_similarity, teacher_similarities)
                # 添加蒸馏损失到总损失
                total_loss = total_loss + self.ensemble_distill_weight * ensemble_distill_loss
                # 记录蒸馏损失
                self.log("train/ensemble_distill_loss", ensemble_distill_loss, batch_size=len(audio_embeddings), sync_dist=True)
                self.log("train/total_loss_with_distill", total_loss, batch_size=len(audio_embeddings), sync_dist=True)
        # 🔥 集成知识蒸馏损失（Kim论文）
        if self.use_ensemble_distillation and len(self.teacher_models) > 0:
            teacher_similarities = []
            device = audio_embeddings.device
            with torch.no_grad():
                for teacher in self.teacher_models:
                    # 确保教师在正确的设备上
                    if next(teacher.parameters()).device != device:
                        teacher = teacher.to(device)
                    # 教师前向传播
                    try:
                        teacher_audio_emb, teacher_text_emb = teacher.forward(batch)
                        teacher_sim = torch.matmul(teacher_audio_emb, teacher_text_emb.T)
                        teacher_similarities.append(teacher_sim)
                    except Exception as e:
                        print(f"⚠️  教师模型前向传播失败: {e}")
            # 如果成功获取教师输出，计算蒸馏损失
            if len(teacher_similarities) > 0:
                ensemble_distill_loss = self.ensemble_distill_loss_fn(student_similarity, teacher_similarities)
                # 添加蒸馏损失到总损失
                total_loss = total_loss + self.ensemble_distill_weight * ensemble_distill_loss
                # 记录蒸馏损失
                self.log("train/ensemble_distill_loss", ensemble_distill_loss, batch_size=len(audio_embeddings), sync_dist=True)
                self.log("train/total_loss_with_distill", total_loss, batch_size=len(audio_embeddings), sync_dist=True)
        # 🔥 在线知识蒸馏损失（使用EMA教师）
        total_loss = contrastive_loss
        if self.use_online_distillation and self.ema_teacher is not None:
            with torch.no_grad():
                teacher_similarity = self.ema_teacher.compute_similarity(batch)
            online_distill_loss = self.online_distill_loss_fn(student_similarity, teacher_similarity)
            total_loss = total_loss + online_distill_loss
            self.log("train/online_distill_loss", online_distill_loss, batch_size=len(audio_embeddings), sync_dist=True)
        # 🔥 聚类引导分类损失
        if self.use_clustering_classification and self.clusterer is not None:
            # 获取聚类标签
            cluster_labels = []
            for caption in [c[0] for c in batch['captions']]:
                cluster_labels.append(self.clusterer.get_cluster_label(caption))
            cluster_labels = torch.tensor(cluster_labels, device=audio_embeddings.device)
            # 计算分类logits
            audio_logits, text_logits = self.clustering_classifier(audio_embeddings, text_embeddings)
            # 计算分类损失
            clustering_loss = self.clustering_classifier.compute_loss(audio_logits, text_logits, cluster_labels)
            total_loss = total_loss + self.clustering_weight * clustering_loss
            self.log("train/clustering_loss", clustering_loss, batch_size=len(audio_embeddings), sync_dist=True)
        # 🔥 AudioCLIP教师蒸馏损失（方案C - 最强优化）
        if self.use_audioclip_distillation and self.audioclip_teacher is not None:
            with torch.no_grad():
                # 使用AudioCLIP教师模型编码特征
                teacher_audio = self.audioclip_teacher.encode_audio(audio_embeddings)
                teacher_text = self.audioclip_teacher.encode_text(text_embeddings)
            # 计算蒸馏损失
            audioclip_distill_loss = self.audioclip_distillation_loss_fn(
                audio_embeddings, 
                text_embeddings,
                teacher_audio,
                teacher_text
            )
            total_loss = total_loss + audioclip_distill_loss
            self.log("train/audioclip_distill_loss", audioclip_distill_loss, batch_size=len(audio_embeddings), sync_dist=True)
        # 🔥 多教师蒸馏损失（CLIP，默认禁用）
        if self.use_multi_teacher_distillation and self.teacher_ensemble is not None:
            with torch.no_grad():
                teacher_similarities = self.teacher_ensemble.compute_teacher_similarities(
                    audio_embeddings,
                    [c[0] for c in batch['captions']]
                )
            if len(teacher_similarities) > 0:
                distill_loss = self.multi_teacher_distillation_loss_fn(
                    student_similarity, 
                    teacher_similarities
                )
                total_loss = total_loss + distill_loss
                self.log("train/distill_loss", distill_loss, batch_size=len(audio_embeddings), sync_dist=True)
        # 🔥 MoE负载均衡损失
        if self.use_cross_attention and hasattr(self, 'audio_to_text_attn') and self.audio_to_text_attn.use_moe:
            moe_load_balance_loss = 0.0
            count = 0
            if hasattr(self.audio_to_text_attn.ffn, 'load_balance_loss'):
                moe_load_balance_loss += self.audio_to_text_attn.ffn.load_balance_loss
                count += 1
            if hasattr(self.text_to_audio_attn.ffn, 'load_balance_loss'):
                moe_load_balance_loss += self.text_to_audio_attn.ffn.load_balance_loss
                count += 1
            if count > 0:
                moe_load_balance_loss = moe_load_balance_loss / count
                total_loss = total_loss + self.moe_load_balance_weight * moe_load_balance_loss
                self.log("train/moe_load_balance_loss", moe_load_balance_loss, batch_size=len(audio_embeddings), sync_dist=True)
        # 检查loss是否异常
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid loss detected at batch {batch_idx}: {total_loss.item()}")
            return None
        self.log("train/loss", total_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        self.log("train/contrastive_loss", contrastive_loss, batch_size=len(audio_embeddings), sync_dist=True)
        self.log('train/tau', torch.abs(self.tau), sync_dist=True)
        # 🔥 更新 EMA Teacher
        if self.use_online_distillation and self.ema_teacher is not None:
            self.ema_teacher.update(self)
        # 更新 EMA 模型
        if self.use_ema and self.ema_model is not None:
            self.update_ema()
        return total_loss
    def compute_loss(self, audio_embeddings, text_embeddings, batch):
        """改进的损失函数，支持多种策略"""
        # 计算相似度矩阵
        C = torch.matmul(audio_embeddings, text_embeddings.T)
        # 限制相似度范围，防止数值不稳定
        C = torch.clamp(C, min=-1.0, max=1.0)
        # 温度缩放，添加下限防止tau过小
        tau = torch.clamp(torch.abs(self.tau), min=0.01, max=1.0)
        C = C / tau
        # 构建标签
        paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
        I = torch.tensor(paths[None, :] == paths[:, None], device=C.device)
        # 🔥 修复：处理 batch_size=1 的情况
        if C.dim() < 2 or C.shape[0] == 1 or C.shape[1] == 1:
            # batch_size=1 或维度不足，返回0损失
            return C.sum() * 0.0
        if self.loss_type == 'improved_infonce':
            # 改进的 InfoNCE 损失，加入 Hard Negative Mining
            # 🔥 使用改进的损失函数
            if hasattr(self, 'improved_loss_fn') and self.use_improved_loss:
                loss, loss_components = self.improved_loss_fn(audio_embeddings, text_embeddings)
                # 记录各个损失分量
                for name, value in loss_components.items():
                    self.log(f'train/loss_{name}', value, batch_size=len(audio_embeddings), sync_dist=True)
            else:
                loss = self.improved_infonce_loss(C, I)
        elif self.loss_type == 'focal':
            # Focal Loss 变体
            loss = self.focal_contrastive_loss(C, I)
        else:
            # 原始损失
            C_audio = torch.log_softmax(C, dim=0)
            C_text = torch.log_softmax(C, dim=1)
            loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
        return loss
    def improved_infonce_loss(self, C, I):
        """改进的 InfoNCE 损失，加入 Hard Negative Mining"""
        batch_size = C.shape[0]
        # 🔥 修复：处理 batch_size=1 的情况
        if batch_size == 1 or C.shape[1] == 1:
            return C.sum() * 0.0
        # 标准的对比学习损失
        # Audio-to-Text 方向
        C_audio = torch.log_softmax(C, dim=0)
        pos_audio = C_audio[torch.where(I)].mean()
        # Text-to-Audio 方向
        C_text = torch.log_softmax(C, dim=1)
        pos_text = C_text[torch.where(I)].mean()
        # 基础损失
        standard_loss = -0.5 * (pos_audio + pos_text)
        # Hard Negative Mining（使用log_softmax后的值，更稳定）
        neg_mask_audio = ~I
        neg_mask_text = ~I
        if neg_mask_audio.any() and neg_mask_text.any():
            # 找到最难的负样本（log概率最高的）
            C_audio_neg = C_audio.clone()
            C_audio_neg[I] = -1e4  # 🔥 修复：使用-1e4而不是-1e9
            hard_neg_audio = C_audio_neg.max(dim=0)[0]
            C_text_neg = C_text.clone()
            C_text_neg[I] = -1e4  # 🔥 修复：使用-1e4而不是-1e9
            hard_neg_text = C_text_neg.max(dim=1)[0]
            # Hard negative loss（惩罚难负样本的高置信度）
            # 使用更温和的权重，避免训练不稳定
            hard_neg_weight = self.hard_negative_weight * min(1.0, self.current_epoch / 10.0)  # 逐渐增加权重
            hard_neg_loss = hard_neg_weight * 0.5 * (hard_neg_audio.mean() + hard_neg_text.mean())
            loss = standard_loss + hard_neg_loss
        else:
            loss = standard_loss
        return loss
    def focal_contrastive_loss(self, C, I):
        """Focal Loss 变体，关注难样本"""
        # 🔥 修复：处理 batch_size=1 的情况
        if C.shape[0] == 1 or C.shape[1] == 1:
            return C.sum() * 0.0
        # Audio-to-Text
        probs_audio = torch.softmax(C, dim=0)
        log_probs_audio = torch.log_softmax(C, dim=0)
        # 计算 focal weight
        focal_weight_audio = (1 - probs_audio[torch.where(I)]) ** self.focal_gamma
        pos_audio = -(focal_weight_audio * log_probs_audio[torch.where(I)]).mean()
        # Text-to-Audio
        probs_text = torch.softmax(C, dim=1)
        log_probs_text = torch.log_softmax(C, dim=1)
        focal_weight_text = (1 - probs_text[torch.where(I)]) ** self.focal_gamma
        pos_text = -(focal_weight_text * log_probs_text[torch.where(I)]).mean()
        loss = 0.5 * (pos_audio + pos_text)
        return loss
    def update_ema(self):
        """更新 EMA 模型参数"""
        if self.ema_model is None:
            # 首次初始化 EMA 模型
            self.ema_model = copy.deepcopy(self.state_dict())
        else:
            # 更新 EMA 参数
            with torch.no_grad():
                for key, value in self.state_dict().items():
                    if key in self.ema_model:
                        self.ema_model[key] = (
                            self.ema_decay * self.ema_model[key] + 
                            (1 - self.ema_decay) * value
                        )
    def validation_step(self, batch, batch_idx):
        # 🔥 修复：不再在验证时切换模型状态，直接使用当前模型
        # EMA 的作用已经在训练过程中体现，验证时使用训练模型即可
        audio_embeddings, text_embeddings = self.forward(batch)
        args = {
            'audio_embeddings': copy.deepcopy(audio_embeddings.detach()),
            'text_embeddings': copy.deepcopy(text_embeddings.detach()),
            'caption': [c[0] for c in batch['captions']],
            'path': batch['fname']
        }
        self.validation_outputs.append(args)
    def on_validation_epoch_end(self, prefix='val'):
        outputs = self.validation_outputs
        # concatenate metadata
        paths = np.array([p for b in outputs for p in b['path']])
        captions = np.array([p for b in outputs for p in b['caption']])
        # audios in clotho can have five captions
        # this snippet discards every occurrence of a duplicate audio
        #
        target = [] # prediction targets for later
        select = [] # indices of the first occurrence for later
        first_occurrence = {} # temporary cache to keep track of first occurrences
        for i, p in enumerate(paths): # iterate over all paths
            index = first_occurrence.get(p)
            if index is None:  # First time seeing this path
                index = len(first_occurrence)
                first_occurrence[p] = index
                select.append(i) # these audios will be selected
            target.append(index) # all paths need a target - choose the correct one
        paths = paths[select]
        # concatenate embeddings
        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select]# only select unique audios
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs])
        # concatenate global ranking
        C = torch.matmul(text_embeddings, audio_embeddings.T)
        # get top 10
        top_ten = C.topk(10, dim=1)[1].detach().cpu().numpy()
        target = np.array(target)
        # recall metrics
        r_1 = (top_ten[:, :1] == target[:, None]).sum(axis=1).mean()
        r_5 = (top_ten[:, :5] == target[:, None]).sum(axis=1).mean()
        r_10 = (top_ten == target[:, None]).sum(axis=1).mean()
        # mAP@10
        AP = 1 / ((top_ten == target[:, None]).argmax(axis=1) + 1)
        AP[~(top_ten == target[:, None]).any(axis=1)] = 0
        mAP = AP.mean()
        # log retrieval performance
        self.log(f'{prefix}/R@1', r_1)
        self.log(f'{prefix}/R@5', r_5)
        self.log(f'{prefix}/R@10', r_10)
        self.log(f'{prefix}/mAP@10', mAP)
        if os.path.exists(f'resources/metadata_eval.csv') and prefix == 'test':
            matched_files = pd.read_csv(f'resources/metadata_eval.csv')
            matched_files["audio_filenames"] = matched_files["audio_filenames"].transform(lambda x: ast.literal_eval(x))
            def get_ranks(c, r):
                ranks = [i.item() for i in torch.argsort(torch.argsort(-c))[r]]
                return ranks
            # index of query in C
            matched_files["query_index"] = matched_files["query"].transform(lambda x: captions.tolist().index(x))
            # new ground truth
            matched_files["new_audio_indices"] = matched_files["audio_filenames"].transform(lambda x: [paths.tolist().index(y) for y in x])
            matched_files["TP_ranks"] = matched_files.apply(lambda row: get_ranks(C[row["query_index"]], row["new_audio_indices"]), axis=1)
            def average_precision_at_k(relevant_ranks, k=10):
                relevant_ranks = sorted(relevant_ranks)
                ap = 0.0
                for i, rank in enumerate(relevant_ranks, start=1):
                    if rank >= k:
                        break
                    ap += i / (rank + 1) # precision at threshold
                return ap / len(relevant_ranks)  # Normalize by total number of relevant items
            new_mAP = matched_files["TP_ranks"].apply(lambda ranks: average_precision_at_k(ranks, 10)).mean()
            self.log(f'{prefix}_multiple_positives/mAP@10', new_mAP)
        # empty cached batches from validation loop
        self.validation_outputs.clear()
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix='test')
    def configure_optimizers(self):
        weight_decay = self.kwargs.get('weight_decay', 0.01)  # 增加默认权重衰减
        # 分层学习率：为不同模块设置不同的学习率
        use_layerwise_lr = self.kwargs.get('use_layerwise_lr', True)
        if use_layerwise_lr:
            # 预训练编码器使用较小学习率，投影层使用较大学习率
            audio_encoder_params = list(self.audio_embedding_model.parameters())
            text_encoder_params = list(self.text_embedding_model.parameters())
            projection_params = list(self.audio_projection.parameters()) + list(self.text_projection.parameters())
            param_groups = [
                {'params': audio_encoder_params, 'lr': self.kwargs['max_lr'] * 0.1, 'weight_decay': weight_decay},
                {'params': text_encoder_params, 'lr': self.kwargs['max_lr'] * 0.1, 'weight_decay': weight_decay},
                {'params': projection_params, 'lr': self.kwargs['max_lr'], 'weight_decay': weight_decay * 0.1},
            ]
            # 如果有交叉注意力模块，也添加进去
            if self.use_cross_attention:
                cross_attn_params = list(self.audio_to_text_attn.parameters()) + list(self.text_to_audio_attn.parameters())
                param_groups.append({'params': cross_attn_params, 'lr': self.kwargs['max_lr'], 'weight_decay': weight_decay * 0.1})
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.98),  # 调整 beta2 以提高稳定性
                eps=1e-6,
                amsgrad=False
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
                amsgrad=False
            )
        return optimizer
    def lr_scheduler_step(self, batch_idx):
        steps_per_epoch = self.trainer.num_training_batches
        min_lr = self.kwargs['min_lr']
        max_lr = self.kwargs['max_lr']
        current_step = self.current_epoch * steps_per_epoch + batch_idx
        warmup_steps = self.kwargs['warmup_epochs'] * steps_per_epoch
        total_steps = (self.kwargs['warmup_epochs'] + self.kwargs['rampdown_epochs']) * steps_per_epoch
        decay_steps = total_steps - warmup_steps
        # 改进的学习率调度策略 - 使用 Cosine Annealing with Warm Restarts
        use_improved_schedule = self.kwargs.get('use_improved_schedule', True)
        use_cosine_restarts = self.kwargs.get('use_cosine_restarts', False)
        if use_improved_schedule:
            # 使用更平滑的 warmup
            if current_step < warmup_steps:
                # 使用平方根 warmup（更平滑）
                progress = current_step / warmup_steps
                lr = min_lr + (max_lr - min_lr) * (progress ** 0.5)
            elif current_step < total_steps:
                if use_cosine_restarts:
                    # Cosine Annealing with Warm Restarts - 每隔一段时间重启学习率
                    restart_period = self.kwargs.get('restart_period', 15) * steps_per_epoch
                    steps_since_warmup = current_step - warmup_steps
                    # 计算当前在哪个重启周期内
                    cycle_progress = (steps_since_warmup % restart_period) / restart_period
                    # 使用 cosine annealing，但在每个周期结束时重启到 max_lr
                    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * cycle_progress))
                else:
                    # 标准 Cosine annealing
                    decay_progress = (current_step - warmup_steps) / decay_steps
                    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:
                lr = min_lr
        else:
            # 原始调度策略
            if current_step < warmup_steps:
                lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
            elif current_step < total_steps:
                decay_progress = (current_step - warmup_steps) / decay_steps
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:
                lr = min_lr
        # 应用学习率到所有参数组
        for i, param_group in enumerate(self.optimizers(use_pl_optimizer=False).param_groups):
            # 如果使用分层学习率，保持相对比例
            if self.kwargs.get('use_layerwise_lr', True) and i < 2:
                # 编码器使用 0.1 倍学习率
                param_group['lr'] = lr * 0.1
            else:
                param_group['lr'] = lr
        self.log('train/lr', lr, sync_dist=True)

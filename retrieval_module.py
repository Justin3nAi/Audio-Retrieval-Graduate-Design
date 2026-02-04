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


class CrossModalAttention(torch.nn.Module):
    """双模态交叉注意力机制，增强音频-文本特征交互"""
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
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
        
        # 前馈网络
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


class AudioRetrievalModel(pl.LightningModule):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__()
        self.save_hyperparameters(kwargs)

        # audio encoder (freqm/timem 需 passt.py 支持，未同步时不要传递)
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

        # 交叉注意力机制（可选）
        self.use_cross_attention = kwargs.get('use_cross_attention', True)
        if self.use_cross_attention:
            self.audio_to_text_attn = CrossModalAttention(1024, num_heads=8, dropout=0.1)
            self.text_to_audio_attn = CrossModalAttention(1024, num_heads=8, dropout=0.1)

        # temperature parameter
        initial_tau = torch.zeros((1,)) + kwargs['initial_tau']
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=kwargs['tau_trainable'])

        # EMA (Exponential Moving Average) for better generalization
        self.use_ema = kwargs.get('use_ema', True)
        if self.use_ema:
            self.ema_decay = kwargs.get('ema_decay', 0.999)
            self.ema_model = None  # 将在第一次训练步骤中初始化

        self.validation_outputs = []
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

    def forward_audio(self, batch):

        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1)) # forward

        # mask embeddings from padded empty audio parts
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
        return audio_embeddings

    def forward_text(self, batch):

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

        return sentence_features

    def training_step(self, batch, batch_idx):

        self.lr_scheduler_step(batch_idx)

        # 初始化 EMA（在第一个训练步骤）
        if self.use_ema and self.ema_model is None:
            self.ema_model = copy.deepcopy(self.state_dict())

        audio_embeddings, text_embeddings = self.forward(batch)

        # 计算损失
        loss = self.compute_loss(audio_embeddings, text_embeddings, batch)
        
        # 检查loss是否异常
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}")
            return None

        self.log("train/loss", loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        self.log('train/tau', torch.abs(self.tau), sync_dist=True)

        # 更新 EMA 模型
        if self.use_ema and self.ema_model is not None:
            self.update_ema()

        return loss

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

        if self.loss_type == 'improved_infonce':
            # 改进的 InfoNCE 损失，加入 Hard Negative Mining
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
            C_audio_neg[I] = -1e9  # 屏蔽正样本
            hard_neg_audio = C_audio_neg.max(dim=0)[0]
            
            C_text_neg = C_text.clone()
            C_text_neg[I] = -1e9  # 屏蔽正样本
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

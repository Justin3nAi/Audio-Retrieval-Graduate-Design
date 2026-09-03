"""

多粒度语义对齐模块 (Multi-Granularity Semantic Alignment)



学术创新点：

1. 同时对齐全局语义（句子级）和局部细节（词/帧级）

2. 使用跨模态注意力捕获细粒度对应关系

3. 层次化的相似度计算策略



预期效果：+3-5% mAP提升

"""



import torch

import torch.nn as nn

import torch.nn.functional as F





class MultiGranularityAlignment(nn.Module):

    """

    多粒度对齐模块

    

    核心思想：

    - 全局对齐：捕获整体语义（如"狗在叫" vs "dog barking"）

    - 局部对齐：捕获细节对应（如"狗叫声" ↔ "dog" + "barking"）

    """

    

    def __init__(

        self, 

        embed_dim=1024,

        num_heads=8,

        dropout=0.1,

        global_weight=0.7,

        local_weight=0.3

    ):

        super().__init__()

        

        self.embed_dim = embed_dim

        self.global_weight = global_weight

        self.local_weight = local_weight

        

        # ============ 全局对齐 ============

        # 简单的投影层，保持全局语义对齐

        self.global_audio_proj = nn.Sequential(

            nn.Linear(embed_dim, embed_dim),

            nn.LayerNorm(embed_dim),

            nn.Dropout(dropout)

        )

        

        self.global_text_proj = nn.Sequential(

            nn.Linear(embed_dim, embed_dim),

            nn.LayerNorm(embed_dim),

            nn.Dropout(dropout)

        )

        

        # ============ 局部对齐 ============

        # 跨模态注意力：音频帧 ↔ 文本词

        self.audio_to_text_attention = nn.MultiheadAttention(

            embed_dim, 

            num_heads=num_heads, 

            dropout=dropout,

            batch_first=True

        )

        

        self.text_to_audio_attention = nn.MultiheadAttention(

            embed_dim,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True

        )

        

        # 局部特征投影

        self.local_audio_proj = nn.Linear(embed_dim, embed_dim)

        self.local_text_proj = nn.Linear(embed_dim, embed_dim)

        

        # ============ 相似度融合 ============

        # 学习如何组合全局和局部相似度

        self.similarity_fusion = nn.Sequential(

            nn.Linear(2, 64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(64, 1)

        )

        

        print(f"✅ MultiGranularityAlignment initialized")

        print(f"   - Embed dim: {embed_dim}")

        print(f"   - Num heads: {num_heads}")

        print(f"   - Global weight: {global_weight}")

        print(f"   - Local weight: {local_weight}")

    

    def compute_global_similarity(self, audio_global, text_global):

        """

        计算全局相似度（句子级）

        

        Args:

            audio_global: (batch, embed_dim) - 全局音频特征

            text_global: (batch, embed_dim) - 全局文本特征

        

        Returns:

            global_sim: (batch, batch) - 全局相似度矩阵

        """

        # 投影到对齐空间

        audio_aligned = self.global_audio_proj(audio_global)

        text_aligned = self.global_text_proj(text_global)

        

        # L2归一化

        audio_aligned = F.normalize(audio_aligned, p=2, dim=-1)

        text_aligned = F.normalize(text_aligned, p=2, dim=-1)

        

        # 计算余弦相似度矩阵

        global_sim = torch.matmul(audio_aligned, text_aligned.T)

        

        return global_sim

    

    def compute_local_similarity(

        self, 

        audio_frames, 

        text_tokens,

        audio_mask=None,

        text_mask=None

    ):

        """

        计算局部相似度（帧/词级）

        

        Args:

            audio_frames: (batch, num_frames, embed_dim) - 音频帧序列

            text_tokens: (batch, num_tokens, embed_dim) - 文本词序列

            audio_mask: (batch, num_frames) - 音频帧mask

            text_mask: (batch, num_tokens) - 文本词mask

        

        Returns:

            local_sim: (batch, batch) - 局部相似度矩阵

        """

        batch_size = audio_frames.shape[0]

        

        # 1. 音频帧 → 文本词的注意力

        # 对每个音频，找到最相关的文本词

        audio_to_text_list = []

        for i in range(batch_size):

            # 当前音频的所有帧 attend to 所有文本的所有词

            audio_query = audio_frames[i:i+1].expand(batch_size, -1, -1)  # (batch, num_frames, dim)

            

            # 注意力机制

            attended_text, attn_weights = self.audio_to_text_attention(

                audio_query,  # query: 当前音频的帧

                text_tokens,  # key: 所有文本的词

                text_tokens,  # value: 所有文本的词

                key_padding_mask=text_mask if text_mask is not None else None

            )

            

            # 聚合：对每个文本，计算与当前音频的匹配度

            # (batch, num_frames, dim) -> (batch, dim)

            if audio_mask is not None:

                mask_expanded = audio_mask[i:i+1].unsqueeze(-1).expand_as(attended_text)

                attended_text = attended_text * mask_expanded

                audio_to_text_feat = attended_text.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

            else:

                audio_to_text_feat = attended_text.mean(dim=1)

            

            audio_to_text_list.append(audio_to_text_feat)

        

        # (batch, batch, dim)

        audio_to_text_matrix = torch.stack(audio_to_text_list, dim=0)

        

        # 2. 文本词 → 音频帧的注意力（对称操作）

        text_to_audio_list = []

        for i in range(batch_size):

            text_query = text_tokens[i:i+1].expand(batch_size, -1, -1)

            

            attended_audio, _ = self.text_to_audio_attention(

                text_query,

                audio_frames,

                audio_frames,

                key_padding_mask=audio_mask if audio_mask is not None else None

            )

            

            if text_mask is not None:

                mask_expanded = text_mask[i:i+1].unsqueeze(-1).expand_as(attended_audio)

                attended_audio = attended_audio * mask_expanded

                text_to_audio_feat = attended_audio.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

            else:

                text_to_audio_feat = attended_audio.mean(dim=1)

            

            text_to_audio_list.append(text_to_audio_feat)

        

        text_to_audio_matrix = torch.stack(text_to_audio_list, dim=0)

        

        # 3. 计算局部相似度

        # 投影

        audio_to_text_proj = self.local_audio_proj(audio_to_text_matrix)

        text_to_audio_proj = self.local_text_proj(text_to_audio_matrix)

        

        # 归一化

        audio_to_text_proj = F.normalize(audio_to_text_proj, p=2, dim=-1)

        text_to_audio_proj = F.normalize(text_to_audio_proj, p=2, dim=-1)

        

        # 双向相似度的平均

        # audio_to_text_proj: (batch_audio, batch_text, dim)

        # 计算对角线相似度

        local_sim = 0

        for i in range(batch_size):

            local_sim += F.cosine_similarity(

                audio_to_text_proj[i], 

                text_to_audio_proj[:, i], 

                dim=-1

            ).unsqueeze(0)

        

        local_sim = local_sim / batch_size

        

        return local_sim

    

    def forward(

        self, 

        audio_global, 

        text_global,

        audio_frames=None,

        text_tokens=None,

        audio_mask=None,

        text_mask=None,

        return_components=False

    ):

        """

        前向传播

        

        Args:

            audio_global: (batch, embed_dim) - 全局音频特征

            text_global: (batch, embed_dim) - 全局文本特征

            audio_frames: (batch, num_frames, embed_dim) - 音频帧序列（可选）

            text_tokens: (batch, num_tokens, embed_dim) - 文本词序列（可选）

            audio_mask: (batch, num_frames) - 音频mask（可选）

            text_mask: (batch, num_tokens) - 文本mask（可选）

            return_components: bool - 是否返回各个组件的相似度

        

        Returns:

            combined_sim: (batch, batch) - 组合后的相似度矩阵

            或 (combined_sim, global_sim, local_sim) 如果 return_components=True

        """

        # 1. 全局相似度（必须）

        global_sim = self.compute_global_similarity(audio_global, text_global)

        

        # 2. 局部相似度（如果提供了帧/词序列）

        if audio_frames is not None and text_tokens is not None:

            local_sim = self.compute_local_similarity(

                audio_frames, 

                text_tokens,

                audio_mask,

                text_mask

            )

            

            # 3. 融合全局和局部相似度

            # 方法1: 固定权重融合

            combined_sim = (

                self.global_weight * global_sim + 

                self.local_weight * local_sim

            )

            

            # 方法2: 学习权重融合（可选，更高级）

            # sim_stack = torch.stack([global_sim, local_sim], dim=-1)  # (batch, batch, 2)

            # combined_sim = self.similarity_fusion(sim_stack).squeeze(-1)

            

        else:

            # 如果没有局部特征，只使用全局相似度

            combined_sim = global_sim

            local_sim = None

        

        if return_components:

            return combined_sim, global_sim, local_sim

        else:

            return combined_sim





class MultiGranularityRetrievalLoss(nn.Module):

    """

    多粒度检索损失

    

    结合全局和局部的对比学习损失

    """

    

    def __init__(self, temperature=0.07, global_weight=0.7, local_weight=0.3):

        super().__init__()

        self.temperature = temperature

        self.global_weight = global_weight

        self.local_weight = local_weight

    

    def compute_contrastive_loss(self, similarity_matrix, labels):

        """

        计算对比学习损失

        

        Args:

            similarity_matrix: (batch, batch) - 相似度矩阵

            labels: (batch, batch) - 标签矩阵（正样本为True）

        

        Returns:

            loss: scalar - 对比损失

        """

        # 温度缩放

        similarity_matrix = similarity_matrix / self.temperature

        

        # 🔥 修复：处理 batch_size=1 的情况

        if similarity_matrix.dim() == 0:

            # 标量，直接返回0损失

            return similarity_matrix.sum() * 0.0

        elif similarity_matrix.dim() == 1:

            # 1维向量，扩展维度

            similarity_matrix = similarity_matrix.unsqueeze(0)

            labels = labels.unsqueeze(0) if labels.dim() == 1 else labels

        

        # 确保至少是2维

        if similarity_matrix.shape[0] == 1 or similarity_matrix.shape[1] == 1:

            # batch_size=1，返回0损失（无法计算对比损失）

            return similarity_matrix.sum() * 0.0

        

        # Audio-to-Text方向

        log_prob_a2t = F.log_softmax(similarity_matrix, dim=1)

        loss_a2t = -log_prob_a2t[labels].mean()

        

        # Text-to-Audio方向

        log_prob_t2a = F.log_softmax(similarity_matrix, dim=0)

        loss_t2a = -log_prob_t2a[labels].mean()

        

        # 双向损失

        loss = 0.5 * (loss_a2t + loss_t2a)

        

        return loss

    

    def forward(self, global_sim, local_sim, labels):

        """

        计算总损失

        

        Args:

            global_sim: (batch, batch) - 全局相似度矩阵

            local_sim: (batch, batch) - 局部相似度矩阵（可选）

            labels: (batch, batch) - 标签矩阵

        

        Returns:

            total_loss: scalar - 总损失

            loss_dict: dict - 各个损失的详细信息

        """

        # 全局损失

        global_loss = self.compute_contrastive_loss(global_sim, labels)

        

        # 局部损失（如果有）

        if local_sim is not None:

            local_loss = self.compute_contrastive_loss(local_sim, labels)

            

            # 加权组合

            total_loss = (

                self.global_weight * global_loss + 

                self.local_weight * local_loss

            )

            

            loss_dict = {

                'total_loss': total_loss.item(),

                'global_loss': global_loss.item(),

                'local_loss': local_loss.item()

            }

        else:

            total_loss = global_loss

            loss_dict = {

                'total_loss': total_loss.item(),

                'global_loss': global_loss.item()

            }

        

        return total_loss, loss_dict





# ============ 辅助函数 ============



def extract_audio_frames(audio_embedding_model, audio, max_frames=10):

    """

    从音频编码器提取帧级特征

    

    Args:

        audio_embedding_model: 音频编码器

        audio: (batch, channels, samples) - 音频输入

        max_frames: int - 最大帧数

    

    Returns:

        audio_frames: (batch, num_frames, embed_dim) - 音频帧特征

        audio_mask: (batch, num_frames) - 帧mask

    """

    # 这里需要根据你的音频编码器实现

    # 示例：假设PaSST返回 (batch, num_segments, embed_dim)

    

    with torch.no_grad():

        audio_features = audio_embedding_model(audio.mean(1))

    

    # 如果是3D张量，直接使用

    if audio_features.dim() == 3:

        batch_size, num_frames, embed_dim = audio_features.shape

        

        # 创建mask（假设所有帧都有效）

        audio_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool, device=audio.device)

        

        return audio_features, audio_mask

    

    # 如果是2D张量，需要扩展

    elif audio_features.dim() == 2:

        batch_size, embed_dim = audio_features.shape

        # 复制为多帧（简化处理）

        audio_frames = audio_features.unsqueeze(1).expand(-1, max_frames, -1)

        audio_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool, device=audio.device)

        

        return audio_frames, audio_mask

    

    else:

        raise ValueError(f"Unexpected audio features shape: {audio_features.shape}")





def extract_text_tokens(text_embedding_model, tokenized_input):

    """

    从文本编码器提取词级特征

    

    Args:

        text_embedding_model: 文本编码器（RoBERTa）

        tokenized_input: dict - tokenizer的输出

    

    Returns:

        text_tokens: (batch, num_tokens, embed_dim) - 文本词特征

        text_mask: (batch, num_tokens) - 词mask

    """

    # RoBERTa输出所有token的hidden states

    outputs = text_embedding_model(

        input_ids=tokenized_input['input_ids'],

        attention_mask=tokenized_input['attention_mask'],

        output_hidden_states=True

    )

    

    # 使用最后一层的hidden states

    text_tokens = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

    

    # mask: padding的位置为True

    text_mask = (tokenized_input['attention_mask'] == 0)

    

    return text_tokens, text_mask





# ============ 使用示例 ============



if __name__ == '__main__':

    """

    测试多粒度对齐模块

    """

    batch_size = 4

    embed_dim = 1024

    num_frames = 10

    num_tokens = 32

    

    # 创建模块

    alignment_module = MultiGranularityAlignment(

        embed_dim=embed_dim,

        num_heads=8,

        dropout=0.1,

        global_weight=0.7,

        local_weight=0.3

    )

    

    # 模拟数据

    audio_global = torch.randn(batch_size, embed_dim)

    text_global = torch.randn(batch_size, embed_dim)

    audio_frames = torch.randn(batch_size, num_frames, embed_dim)

    text_tokens = torch.randn(batch_size, num_tokens, embed_dim)

    

    # 前向传播

    combined_sim, global_sim, local_sim = alignment_module(

        audio_global,

        text_global,

        audio_frames,

        text_tokens,

        return_components=True

    )

    

    print(f"Combined similarity shape: {combined_sim.shape}")

    print(f"Global similarity shape: {global_sim.shape}")

    print(f"Local similarity shape: {local_sim.shape}")

    

    # 测试损失

    labels = torch.eye(batch_size, dtype=torch.bool)

    loss_fn = MultiGranularityRetrievalLoss(temperature=0.07)

    

    total_loss, loss_dict = loss_fn(global_sim, local_sim, labels)

    print(f"\nLoss: {loss_dict}")


"""
简化的MoE (Mixture of Experts) 模块
用于CrossModalAttention中的FFN层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoEFFN(nn.Module):
    """
    简化的MoE前馈网络
    使用多个专家网络和门控机制
    """
    
    def __init__(self, embed_dim=1024, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络：决定使用哪些专家
        self.gate = nn.Linear(embed_dim, num_experts)
        
        # 专家网络：多个独立的FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        
        Returns:
            output: (batch, seq_len, embed_dim)
            load_balance_loss: 负载均衡损失
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. 计算门控分数
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # 2. 选择top-k个专家
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)  # 重新归一化
        
        # 3. 计算专家输出
        output = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]  # (batch, seq_len)
            expert_weight = top_k_scores[:, :, i:i+1]  # (batch, seq_len, 1)
            
            # 对每个专家计算输出
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id).unsqueeze(-1)  # (batch, seq_len, 1)
                if mask.any():
                    expert_output = self.experts[expert_id](x)
                    output += expert_output * expert_weight * mask.float()
        
        # 4. 计算负载均衡损失
        # 鼓励所有专家被均匀使用
        expert_usage = gate_scores.mean(dim=[0, 1])  # (num_experts,)
        load_balance_loss = self.num_experts * (expert_usage ** 2).sum()
        
        return output, load_balance_loss


class StandardFFN(nn.Module):
    """
    标准的前馈网络（不使用MoE时的备选）
    """
    
    def __init__(self, embed_dim=1024, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        
        Returns:
            output: (batch, seq_len, embed_dim)
            load_balance_loss: 0 (为了接口一致)
        """
        output = self.ffn(x)
        load_balance_loss = torch.tensor(0.0, device=x.device)
        return output, load_balance_loss


# 使用示例
if __name__ == '__main__':
    # 测试MoE FFN
    batch_size = 4
    seq_len = 10
    embed_dim = 1024
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # MoE版本
    moe_ffn = SimpleMoEFFN(embed_dim=embed_dim, num_experts=4, top_k=2)
    output, loss = moe_ffn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Load balance loss: {loss.item():.4f}")
    
    # 标准FFN版本
    std_ffn = StandardFFN(embed_dim=embed_dim)
    output2, loss2 = std_ffn(x)
    
    print(f"\nStandard FFN output shape: {output2.shape}")
    print(f"Standard FFN loss: {loss2.item():.4f}")


























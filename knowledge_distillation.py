"""
知识蒸馏模块 - 改进版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleDistillationLoss(nn.Module):
    """
    集成知识蒸馏损失 - 改进版
    
    关键改进：
    1. 使用 MSE 对齐相似度矩阵（而不是 KL 散度）
    2. 归一化相似度矩阵后再计算损失
    3. 更稳定的训练
    """
    
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_sim, teacher_sims):
        """
        Args:
            student_sim: (batch, batch) - 学生模型的相似度矩阵
            teacher_sims: list of (batch, batch) - 教师模型的相似度矩阵列表
        
        Returns:
            loss: scalar - 蒸馏损失
        """
        # 教师集成：平均相似度矩阵
        teacher_avg = sum(teacher_sims) / len(teacher_sims)
        
        # 温度缩放
        teacher_avg_scaled = teacher_avg / self.temperature
        student_sim_scaled = student_sim / self.temperature
        
        # 方法1：KL 散度（原来的方法）
        # Audio-to-Text
        soft_labels_a2t = F.softmax(teacher_avg_scaled, dim=1)
        student_log_probs_a2t = F.log_softmax(student_sim_scaled, dim=1)
        loss_a2t = F.kl_div(student_log_probs_a2t, soft_labels_a2t, reduction='batchmean')
        
        # Text-to-Audio
        soft_labels_t2a = F.softmax(teacher_avg_scaled, dim=0)
        student_log_probs_t2a = F.log_softmax(student_sim_scaled, dim=0)
        loss_t2a = F.kl_div(student_log_probs_t2a, soft_labels_t2a, reduction='batchmean')
        
        kl_loss = (loss_a2t + loss_t2a) / 2 * (self.temperature ** 2)
        
        # 方法2：MSE 损失（更直接）
        # 归一化到 [0, 1] 范围
        teacher_norm = F.softmax(teacher_avg_scaled, dim=1)
        student_norm = F.softmax(student_sim_scaled, dim=1)
        mse_loss = F.mse_loss(student_norm, teacher_norm)
        
        # 组合两种损失
        loss = 0.5 * kl_loss + 0.5 * mse_loss
        
        return loss


class SimpleDistillationLoss(nn.Module):
    """
    简单的知识蒸馏损失（单个教师）
    """
    
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_sim, teacher_sim):
        teacher_sim_scaled = teacher_sim / self.temperature
        student_sim_scaled = student_sim / self.temperature
        
        soft_labels_a2t = F.softmax(teacher_sim_scaled, dim=1)
        student_log_probs_a2t = F.log_softmax(student_sim_scaled, dim=1)
        loss_a2t = F.kl_div(student_log_probs_a2t, soft_labels_a2t, reduction='batchmean')
        
        soft_labels_t2a = F.softmax(teacher_sim_scaled, dim=0)
        student_log_probs_t2a = F.log_softmax(student_sim_scaled, dim=0)
        loss_t2a = F.kl_div(student_log_probs_t2a, soft_labels_t2a, reduction='batchmean')
        
        loss = (loss_a2t + loss_t2a) / 2
        return loss * (self.temperature ** 2)


def test_distillation_loss():
    print("Testing distillation loss...")
    batch_size = 8
    student_sim = torch.randn(batch_size, batch_size)
    teacher_sim1 = torch.randn(batch_size, batch_size)
    teacher_sim2 = torch.randn(batch_size, batch_size)
    
    loss_fn = EnsembleDistillationLoss(temperature=1.0)
    loss = loss_fn(student_sim, [teacher_sim1, teacher_sim2])
    print(f"Ensemble distillation loss: {loss.item():.4f}")
    print("SUCCESS")


if __name__ == '__main__':
    test_distillation_loss()

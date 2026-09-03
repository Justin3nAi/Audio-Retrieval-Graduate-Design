"""
Improved Loss Functions for Audio-Text Retrieval
Implements Focal Loss and Hard Negative Mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalContrastiveLoss(nn.Module):
    """
    Focal Loss for Contrastive Learning
    Focuses on hard negatives by down-weighting easy examples
    
    Based on: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, temperature=0.07, gamma=2.0, alpha=0.25):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma  # focusing parameter
        self.alpha = alpha  # balancing parameter
    
    def forward(self, audio_emb, text_emb):
        """
        Args:
            audio_emb: (batch, dim)
            text_emb: (batch, dim)
        Returns:
            focal contrastive loss
        """
        # Normalize embeddings
        audio_emb = F.normalize(audio_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(audio_emb, text_emb.T) / self.temperature
        
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Audio-to-text
        log_probs_a2t = F.log_softmax(sim_matrix, dim=1)
        probs_a2t = F.softmax(sim_matrix, dim=1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight_a2t = (1 - probs_a2t[range(batch_size), labels]) ** self.gamma
        loss_a2t = -self.alpha * focal_weight_a2t * log_probs_a2t[range(batch_size), labels]
        
        # Text-to-audio
        log_probs_t2a = F.log_softmax(sim_matrix.T, dim=1)
        probs_t2a = F.softmax(sim_matrix.T, dim=1)
        
        focal_weight_t2a = (1 - probs_t2a[range(batch_size), labels]) ** self.gamma
        loss_t2a = -self.alpha * focal_weight_t2a * log_probs_t2a[range(batch_size), labels]
        
        return (loss_a2t.mean() + loss_t2a.mean()) / 2


class HardNegativeContrastiveLoss(nn.Module):
    """
    Contrastive Loss with Hard Negative Mining
    Focuses on the hardest negatives in each batch
    """
    
    def __init__(self, temperature=0.07, hard_ratio=0.5):
        super().__init__()
        self.temperature = temperature
        self.hard_ratio = hard_ratio  # ratio of hard negatives to use
    
    def forward(self, audio_emb, text_emb):
        """
        Args:
            audio_emb: (batch, dim)
            text_emb: (batch, dim)
        Returns:
            hard negative contrastive loss
        """
        # Normalize
        audio_emb = F.normalize(audio_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(audio_emb, text_emb.T) / self.temperature
        
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Audio-to-text with hard negative mining
        loss_a2t = self._hard_negative_loss(sim_matrix, labels)
        
        # Text-to-audio with hard negative mining
        loss_t2a = self._hard_negative_loss(sim_matrix.T, labels)
        
        return (loss_a2t + loss_t2a) / 2
    
    def _hard_negative_loss(self, sim_matrix, labels):
        """Compute loss with hard negative mining"""
        batch_size = sim_matrix.size(0)
        
        # Get positive similarities
        pos_sim = sim_matrix[range(batch_size), labels]
        
        # Mask out positives
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        mask[range(batch_size), labels] = False
        
        # Get negative similarities
        neg_sim = sim_matrix[mask].view(batch_size, -1)
        
        # Select hard negatives (top-k highest similarities)
        num_hard = max(1, int(neg_sim.size(1) * self.hard_ratio))
        hard_neg_sim, _ = torch.topk(neg_sim, k=num_hard, dim=1)
        
        # Compute loss with hard negatives
        # log(exp(pos) / (exp(pos) + sum(exp(hard_neg))))
        pos_exp = torch.exp(pos_sim).unsqueeze(1)
        hard_neg_exp = torch.exp(hard_neg_sim)
        
        loss = -torch.log(pos_exp / (pos_exp + hard_neg_exp.sum(dim=1, keepdim=True)))
        
        return loss.mean()


class ImprovedInfoNCELoss(nn.Module):
    """
    Improved InfoNCE with multiple enhancements:
    1. Temperature scaling
    2. Label smoothing
    3. Symmetric loss
    """
    
    def __init__(self, temperature=0.07, label_smoothing=0.1):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(self, audio_emb, text_emb):
        """
        Args:
            audio_emb: (batch, dim)
            text_emb: (batch, dim)
        Returns:
            improved InfoNCE loss
        """
        # Normalize
        audio_emb = F.normalize(audio_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(audio_emb, text_emb.T) / self.temperature
        
        batch_size = sim_matrix.size(0)
        
        # Create smooth labels
        labels = torch.arange(batch_size, device=sim_matrix.device)
        smooth_labels = torch.full_like(sim_matrix, self.label_smoothing / (batch_size - 1))
        smooth_labels[range(batch_size), labels] = 1 - self.label_smoothing
        
        # Audio-to-text
        log_probs_a2t = F.log_softmax(sim_matrix, dim=1)
        loss_a2t = -(smooth_labels * log_probs_a2t).sum(dim=1).mean()
        
        # Text-to-audio
        log_probs_t2a = F.log_softmax(sim_matrix.T, dim=1)
        loss_t2a = -(smooth_labels.T * log_probs_t2a).sum(dim=1).mean()
        
        return (loss_a2t + loss_t2a) / 2


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with weights
    """
    
    def __init__(
        self,
        use_focal=True,
        use_hard_negative=True,
        use_infonce=True,
        focal_weight=0.5,
        hard_neg_weight=0.3,
        infonce_weight=0.2,
        **kwargs
    ):
        super().__init__()
        
        self.use_focal = use_focal
        self.use_hard_negative = use_hard_negative
        self.use_infonce = use_infonce
        
        self.focal_weight = focal_weight
        self.hard_neg_weight = hard_neg_weight
        self.infonce_weight = infonce_weight
        
        if use_focal:
            self.focal_loss = FocalContrastiveLoss(**kwargs.get('focal', {}))
        
        if use_hard_negative:
            self.hard_neg_loss = HardNegativeContrastiveLoss(**kwargs.get('hard_neg', {}))
        
        if use_infonce:
            self.infonce_loss = ImprovedInfoNCELoss(**kwargs.get('infonce', {}))
    
    def forward(self, audio_emb, text_emb):
        """
        Compute combined loss
        
        Args:
            audio_emb: (batch, dim)
            text_emb: (batch, dim)
        Returns:
            combined loss, dict of individual losses
        """
        total_loss = 0
        loss_dict = {}
        
        if self.use_focal:
            focal = self.focal_loss(audio_emb, text_emb)
            total_loss += self.focal_weight * focal
            loss_dict['focal'] = focal.item()
        
        if self.use_hard_negative:
            hard_neg = self.hard_neg_loss(audio_emb, text_emb)
            total_loss += self.hard_neg_weight * hard_neg
            loss_dict['hard_neg'] = hard_neg.item()
        
        if self.use_infonce:
            infonce = self.infonce_loss(audio_emb, text_emb)
            total_loss += self.infonce_weight * infonce
            loss_dict['infonce'] = infonce.item()
        
        return total_loss, loss_dict


def test_losses():
    """Test loss functions"""
    print("Testing improved loss functions...")
    
    batch_size = 32
    dim = 1024
    
    audio_emb = torch.randn(batch_size, dim)
    text_emb = torch.randn(batch_size, dim)
    
    # Test Focal Loss
    focal_loss = FocalContrastiveLoss()
    loss = focal_loss(audio_emb, text_emb)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test Hard Negative Loss
    hard_neg_loss = HardNegativeContrastiveLoss()
    loss = hard_neg_loss(audio_emb, text_emb)
    print(f"Hard Negative Loss: {loss.item():.4f}")
    
    # Test Improved InfoNCE
    infonce_loss = ImprovedInfoNCELoss()
    loss = infonce_loss(audio_emb, text_emb)
    print(f"Improved InfoNCE Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    loss, loss_dict = combined_loss(audio_emb, text_emb)
    print(f"Combined Loss: {loss.item():.4f}")
    print(f"  Components: {loss_dict}")
    
    print("All tests passed!")


if __name__ == '__main__':
    test_losses()

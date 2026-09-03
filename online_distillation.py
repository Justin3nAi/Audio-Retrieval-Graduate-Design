"""
Online distillation placeholder module.
Provides stub classes for loading legacy checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class OnlineDistillationLoss(nn.Module):
    """Placeholder for legacy checkpoint compatibility."""
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_sim, teacher_sim):
        return F.kl_div(
            F.log_softmax(student_sim / self.temperature, dim=1),
            F.softmax(teacher_sim / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)


class EMATeacher:
    """Exponential Moving Average teacher for online distillation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student):
        for ema_p, s_p in zip(self.model.parameters(), student.parameters()):
            ema_p.data.mul_(self.decay).add_(s_p.data, alpha=1 - self.decay)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

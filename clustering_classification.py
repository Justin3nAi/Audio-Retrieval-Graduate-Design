"""
Clustering classification placeholder module.
Provides stub classes for loading legacy checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusteringClassificationLoss(nn.Module):
    """Placeholder for legacy checkpoint compatibility."""
    def __init__(self, num_clusters=20, weight=0.05, **kwargs):
        super().__init__()
        self.num_clusters = num_clusters
        self.weight = weight

    def forward(self, audio_logits, text_logits, cluster_labels):
        loss_a = F.cross_entropy(audio_logits, cluster_labels)
        loss_t = F.cross_entropy(text_logits, cluster_labels)
        return self.weight * (loss_a + loss_t)


class AudioClusterClassifier(nn.Module):
    """Placeholder audio cluster classification head."""
    def __init__(self, input_dim=1024, num_clusters=20, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_clusters)

    def forward(self, x):
        return self.fc(x)


class TextClusterClassifier(nn.Module):
    """Placeholder text cluster classification head."""
    def __init__(self, input_dim=1024, num_clusters=20, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_clusters)

    def forward(self, x):
        return self.fc(x)

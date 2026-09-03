"""
Audio Data Augmentation Module
Implements SpecAugment and other augmentation techniques
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import random


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    https://arxiv.org/abs/1904.08779
    
    Applies time and frequency masking to spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param=30,
        time_mask_param=40,
        n_freq_masks=2,
        n_time_masks=2,
        p=0.5
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p  # probability of applying augmentation
        
        self.freq_masking = T.FrequencyMasking(freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param)
    
    def forward(self, spec):
        """
        Args:
            spec: (batch, freq, time) or (batch, channels, freq, time)
        Returns:
            augmented spec with same shape
        """
        if random.random() > self.p:
            return spec
        
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            spec = self.freq_masking(spec)
        
        # Apply time masking
        for _ in range(self.n_time_masks):
            spec = self.time_masking(spec)
        
        return spec


class AudioMixup(nn.Module):
    """
    Mixup augmentation for audio
    https://arxiv.org/abs/1710.09412
    
    Mixes two audio samples with a random weight.
    """
    
    def __init__(self, alpha=0.2, p=0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p
    
    def forward(self, audio, labels=None):
        """
        Args:
            audio: (batch, channels, samples)
            labels: optional labels for mixing
        Returns:
            mixed audio, (optional) mixed labels
        """
        if random.random() > self.p or audio.size(0) < 2:
            return (audio, labels) if labels is not None else audio
        
        batch_size = audio.size(0)
        
        # Sample mixing weight from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # Random permutation
        index = torch.randperm(batch_size, device=audio.device)
        
        # Mix audio
        mixed_audio = lam * audio + (1 - lam) * audio[index]
        
        if labels is not None:
            mixed_labels = lam * labels + (1 - lam) * labels[index]
            return mixed_audio, mixed_labels
        
        return mixed_audio


class TimeStretch(nn.Module):
    """
    Time stretching augmentation
    Randomly speeds up or slows down audio
    """
    
    def __init__(self, rate_range=(0.8, 1.2), p=0.3):
        super().__init__()
        self.rate_range = rate_range
        self.p = p
    
    def forward(self, audio):
        """
        Args:
            audio: (batch, channels, samples)
        Returns:
            time-stretched audio
        """
        if random.random() > self.p:
            return audio
        
        rate = random.uniform(*self.rate_range)
        
        # Simple resampling-based time stretch
        # For production, use librosa.effects.time_stretch
        target_length = int(audio.size(-1) / rate)
        stretched = torch.nn.functional.interpolate(
            audio,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        # Pad or crop to original length
        if stretched.size(-1) < audio.size(-1):
            pad = audio.size(-1) - stretched.size(-1)
            stretched = torch.nn.functional.pad(stretched, (0, pad))
        else:
            stretched = stretched[..., :audio.size(-1)]
        
        return stretched


class GaussianNoise(nn.Module):
    """
    Add Gaussian noise to audio
    """
    
    def __init__(self, std=0.01, p=0.3):
        super().__init__()
        self.std = std
        self.p = p
    
    def forward(self, audio):
        """
        Args:
            audio: (batch, channels, samples)
        Returns:
            noisy audio
        """
        if random.random() > self.p:
            return audio
        
        noise = torch.randn_like(audio) * self.std
        return audio + noise


class AudioAugmentation(nn.Module):
    """
    Combined audio augmentation pipeline
    """
    
    def __init__(
        self,
        use_spec_augment=True,
        use_mixup=True,
        use_time_stretch=False,
        use_noise=False,
        **kwargs
    ):
        super().__init__()
        
        self.use_spec_augment = use_spec_augment
        self.use_mixup = use_mixup
        self.use_time_stretch = use_time_stretch
        self.use_noise = use_noise
        
        if use_spec_augment:
            self.spec_augment = SpecAugment(**kwargs.get('spec_augment', {}))
        
        if use_mixup:
            self.mixup = AudioMixup(**kwargs.get('mixup', {}))
        
        if use_time_stretch:
            self.time_stretch = TimeStretch(**kwargs.get('time_stretch', {}))
        
        if use_noise:
            self.noise = GaussianNoise(**kwargs.get('noise', {}))
    
    def forward(self, audio, spec=None, labels=None):
        """
        Apply augmentations
        
        Args:
            audio: raw audio (batch, channels, samples)
            spec: spectrogram (batch, freq, time) - optional
            labels: labels for mixup - optional
        
        Returns:
            augmented audio, (optional) augmented spec, (optional) mixed labels
        """
        # Audio-level augmentations
        if self.use_time_stretch:
            audio = self.time_stretch(audio)
        
        if self.use_noise:
            audio = self.noise(audio)
        
        if self.use_mixup:
            if labels is not None:
                audio, labels = self.mixup(audio, labels)
            else:
                audio = self.mixup(audio)
        
        # Spectrogram-level augmentations
        if spec is not None and self.use_spec_augment:
            spec = self.spec_augment(spec)
        
        if labels is not None:
            return audio, spec, labels
        elif spec is not None:
            return audio, spec
        else:
            return audio


def test_augmentation():
    """Test augmentation pipeline"""
    print("Testing audio augmentation...")
    
    # Create dummy data
    batch_size = 4
    audio = torch.randn(batch_size, 1, 160000)  # 10s @ 16kHz
    spec = torch.randn(batch_size, 128, 1000)   # mel spectrogram
    
    # Test SpecAugment
    spec_aug = SpecAugment()
    aug_spec = spec_aug(spec)
    print(f"SpecAugment: {spec.shape} -> {aug_spec.shape}")
    
    # Test Mixup
    mixup = AudioMixup()
    mixed_audio = mixup(audio)
    print(f"Mixup: {audio.shape} -> {mixed_audio.shape}")
    
    # Test full pipeline
    aug_pipeline = AudioAugmentation(
        use_spec_augment=True,
        use_mixup=True,
        use_time_stretch=True,
        use_noise=True
    )
    aug_audio, aug_spec = aug_pipeline(audio, spec)
    print(f"Full pipeline: audio {audio.shape} -> {aug_audio.shape}")
    print(f"               spec {spec.shape} -> {aug_spec.shape}")
    
    print("All tests passed!")


if __name__ == '__main__':
    test_augmentation()

"""
Data augmentation and loss functions for SimCLR-based EEG analysis
"""
import torch
import torch.nn as nn
import numpy as np
from pytorch_metric_learning import losses
from config import (
    GAUSSIAN_NOISE_FACTOR, CHANNEL_DROPOUT_PROB, TIME_SHIFT_RANGE,
    AMPLITUDE_SCALE_RANGE, FREQUENCY_MASK_PROB, FREQUENCY_MASK_RATIO,
    TIME_MASK_PROB, TIME_MASK_RATIO
)


class SimCLR_EEG_Augmentations:
    """최적화된 SimCLR EEG 데이터 증강 클래스"""
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def gaussian_noise(self, x, noise_factor=GAUSSIAN_NOISE_FACTOR):
        """가우시안 노이즈 추가 (인플레이스 연산)"""
        noise = torch.randn_like(x, device=x.device) * noise_factor
        return x + noise
    
    def channel_dropout(self, x, drop_prob=CHANNEL_DROPOUT_PROB):
        """채널 드롭아웃 (GPU 연산 최적화)"""
        if torch.rand(1).item() < drop_prob:
            n_channels = x.shape[0]
            n_drop = max(1, int(n_channels * 0.1))
            drop_mask = torch.ones(n_channels, device=x.device)
            drop_indices = torch.randperm(n_channels, device=x.device)[:n_drop]
            drop_mask[drop_indices] = 0
            return x * drop_mask.unsqueeze(-1)
        return x
    
    def time_shift(self, x, shift_range=TIME_SHIFT_RANGE):
        """시간 축 이동 (GPU 연산 최적화)"""
        max_shift = int(x.shape[1] * shift_range)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift > 0:
            return torch.cat([torch.zeros(x.shape[0], shift, device=x.device), x[:, :-shift]], dim=1)
        elif shift < 0:
            return torch.cat([x[:, -shift:], torch.zeros(x.shape[0], -shift, device=x.device)], dim=1)
        return x
    
    def amplitude_scaling(self, x, scale_range=AMPLITUDE_SCALE_RANGE):
        """진폭 스케일링 (GPU 연산)"""
        scale_factor = torch.rand(1, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return x * scale_factor
    
    def frequency_mask(self, x, mask_prob=FREQUENCY_MASK_PROB, mask_ratio=FREQUENCY_MASK_RATIO):
        """주파수 도메인 마스킹 (GPU 연산 최적화)"""
        if torch.rand(1).item() < mask_prob:
            x_fft = torch.fft.fft(x, dim=1)
            freq_len = x_fft.shape[1]
            mask_len = int(freq_len * mask_ratio)
            start_idx = torch.randint(0, freq_len - mask_len + 1, (1,)).item()
            x_fft[:, start_idx:start_idx + mask_len] = 0
            return torch.fft.ifft(x_fft, dim=1).real
        return x
    
    def time_masking(self, x, mask_prob=TIME_MASK_PROB, mask_ratio=TIME_MASK_RATIO):
        """시간 도메인 마스킹 (GPU 연산 최적화)"""
        if torch.rand(1).item() < mask_prob:
            seq_len = x.shape[1]
            mask_len = int(seq_len * mask_ratio)
            start_idx = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
            x[:, start_idx:start_idx + mask_len] = 0
        return x
    
    def __call__(self, x):
        """최적화된 두 개의 증강된 뷰 생성"""
        # 첫 번째 뷰 (약한 증강)
        view1 = self.gaussian_noise(x, noise_factor=0.03)
        # 두 번째 뷰 (강한 증강) - 원본 텐서 재사용
        view2 = x.clone()
        view2 = self.frequency_mask(view2, mask_prob=0.2)
        return view1, view2


class OptimizedAdaptiveMultiHead_NTXentLoss(nn.Module):
    """최적화된 적응적 다중 헤드 NTXent 손실"""
    def __init__(self, base_temperature=0.1, regularization_weight=0.01):
        super(OptimizedAdaptiveMultiHead_NTXentLoss, self).__init__()
        self.base_temperature = base_temperature
        self.regularization_weight = regularization_weight
        self.frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        self.ntxent_losses = nn.ModuleDict()
        for band in self.frequency_bands:
            self.ntxent_losses[band] = losses.NTXentLoss(temperature=base_temperature)
    
    def compute_regularization_term(self, temperature, feature_dim):
        """
        Temperature regularizer (Algorithm 1, line 15):

            Omega(tau_b) = mean_n [ (d / 2) * log(tau_{b,n}) + 1 / tau_{b,n} ]

        tau_{b,n} in [TEMP_MIN, TEMP_MAX] is the adaptive temperature predicted for sample pair n in
        band b, and d is the projection dimension (128). 1/tau penalizes very small temperatures
        (over-sharp similarity distributions), while (d/2) log tau penalizes very large ones. Its unconstrained
        minimum is tau* = 2/d, so with d = 128 it pushes tau towards the lower bound TEMP_MIN.
        """
        temp_clipped = torch.clamp(temperature, min=1e-8, max=1e8)
        log_term = (feature_dim / 2.0) * torch.log(temp_clipped)
        inv_term = 1.0 / temp_clipped
        omega = log_term + inv_term
        return omega.mean()
    
    def forward(self, projections_i, projections_j, adaptive_temperatures):
        """
        L = sum_b NTXent_b(z_i^b, z_j^b; mean_n tau_{b,n}) + lambda * sum_b Omega(tau_b)

        The per-band NT-Xent uses the batch-mean adaptive temperature as a constant (.item()), so the
        temperature networks receive gradients only through Omega; lambda = regularization_weight.
        Returns (total loss, summed contrastive loss, summed regularization term).
        """
        device = next(iter(projections_i.values())).device
        total_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        num_valid_heads = 0
        
        for band in self.frequency_bands:
            if band in projections_i and band in projections_j:
                try:
                    batch_size = projections_i[band].shape[0]
                    labels = torch.arange(batch_size * 2, device=device)
                    labels[batch_size:] = labels[:batch_size]
                    
                    combined_projections = torch.cat([projections_i[band], projections_j[band]], dim=0)
                    
                    if band in adaptive_temperatures:
                        avg_temp = adaptive_temperatures[band].mean().item()
                        self.ntxent_losses[band].temperature = avg_temp
                    
                    loss = self.ntxent_losses[band](combined_projections, labels)
                    
                    if band in adaptive_temperatures:
                        feature_dim = projections_i[band].shape[1]
                        reg_loss = self.compute_regularization_term(
                            adaptive_temperatures[band], feature_dim
                        )
                        total_reg_loss += reg_loss
                    
                    total_loss += loss
                    num_valid_heads += 1
                    
                except Exception as e:
                    print(f"Error in {band} band: {e}")
                    continue
        
        if num_valid_heads > 0:
            avg_loss = total_loss
            avg_reg_loss = total_reg_loss
            final_loss = avg_loss + self.regularization_weight * avg_reg_loss
            return final_loss, avg_loss, avg_reg_loss
        else:
            return (torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))

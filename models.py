"""
Frequency band extraction and model architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from config import FREQUENCY_BANDS, N_CHANNELS, SIMCLR_HIDDEN_DIM, SIMCLR_OUTPUT_DIM, TEMP_MIN, TEMP_MAX


class ScipyEEG_FrequencyBandExtractor(nn.Module):
    """Scipy 기반 EEG 주파수 밴드별 특징 추출"""
    def __init__(self, sampling_rate=500, n_channels=N_CHANNELS):
        super(ScipyEEG_FrequencyBandExtractor, self).__init__()
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.frequency_bands = FREQUENCY_BANDS
        
        # 각 밴드별 필터 파라미터 미리 계산
        self.filter_coeffs = {}
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            sos = signal.butter(6, [low_freq, high_freq],
                                btype='bandpass',
                                fs=sampling_rate,
                                output='sos')
            self.filter_coeffs[band_name] = sos
        
        # 각 밴드별 후처리 네트워크
        self.band_filters = nn.ModuleDict()
        for band_name in self.frequency_bands.keys():
            self.band_filters[band_name] = nn.Sequential(
                nn.Conv1d(n_channels, n_channels, kernel_size=7, padding=3, groups=n_channels),
                nn.BatchNorm1d(n_channels),
                nn.ReLU(inplace=True)
            )
    
    def apply_scipy_bandpass_filter(self, x, band_name):
        """Scipy 기반 밴드패스 필터 적용"""
        x_np = x.cpu().detach().numpy()
        filtered_signals = []
        for ch in range(x_np.shape[0]):
            try:
                filtered_signal = signal.sosfilt(self.filter_coeffs[band_name], x_np[ch])
                filtered_signals.append(filtered_signal)
            except Exception as e:
                print(f"Filter error for channel {ch}, band {band_name}: {e}")
                filtered_signals.append(x_np[ch])
        
        filtered_tensor = torch.from_numpy(np.array(filtered_signals)).float().to(x.device)
        return filtered_tensor
    
    def forward(self, x):
        """Scipy 기반 주파수 밴드별 특징 추출"""
        band_features = {}
        for band_name in self.frequency_bands.keys():
            filtered_signal = self.apply_scipy_bandpass_filter(x, band_name)
            band_feature = self.band_filters[band_name](filtered_signal)
            band_features[band_name] = band_feature
        return band_features


class OptimizedMultiFrequencyProjectionHead(nn.Module):
    """최적화된 다중 주파수 프로젝션 헤드"""
    def __init__(self, input_dim, hidden_dim=SIMCLR_HIDDEN_DIM, output_dim=SIMCLR_OUTPUT_DIM,
                 temp_min=TEMP_MIN, temp_max=TEMP_MAX):
        super(OptimizedMultiFrequencyProjectionHead, self).__init__()
        self.frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.temp_min = temp_min
        self.temp_max = temp_max
        
        self.projection_heads = nn.ModuleDict()
        for band in self.frequency_bands:
            self.projection_heads[band] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        
        self.temperature_feature_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.temperature_nets = nn.ModuleDict()
        for band in self.frequency_bands:
            self.temperature_nets[band] = nn.Sequential(
                nn.Linear(output_dim * 2, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, 1)
            )
    
    def compute_adaptive_temperature(self, z_i, z_j, band):
        """최적화된 적응적 온도 계산"""
        phi_z_i = self.temperature_feature_net(z_i)
        phi_z_j = self.temperature_feature_net(z_j)
        combined_features = torch.cat([phi_z_i, phi_z_j], dim=1)
        temp_logits = self.temperature_nets[band](combined_features).squeeze(-1)
        temp_sigmoid = torch.sigmoid(temp_logits)
        adaptive_temp = self.temp_min + temp_sigmoid * (self.temp_max - self.temp_min)
        return adaptive_temp
    
    def forward(self, band_features, compute_temperature=False, view_pair=None):
        """최적화된 다중 헤드 프로젝션"""
        projections = {}
        for band in self.frequency_bands:
            if band in band_features:
                pooled_feature = F.adaptive_avg_pool1d(band_features[band], 1).squeeze(-1)
                projection = self.projection_heads[band](pooled_feature)
                projections[band] = F.normalize(projection, p=2, dim=1)
        
        if compute_temperature and view_pair is not None:
            view1_proj, view2_proj = view_pair
            adaptive_temps = {}
            for band in self.frequency_bands:
                if band in projections and band in view1_proj and band in view2_proj:
                    adaptive_temp = self.compute_adaptive_temperature(
                        view1_proj[band], view2_proj[band], band
                    )
                    adaptive_temps[band] = adaptive_temp
            return projections, adaptive_temps
        
        return projections


class ScipyMultiFrequency_SimCLR_Encoder(nn.Module):
    """Scipy 기반 다중 주파수 SimCLR 인코더"""
    def __init__(self, n_channels=N_CHANNELS, sampling_rate=500):
        super(ScipyMultiFrequency_SimCLR_Encoder, self).__init__()
        self.frequency_extractor = ScipyEEG_FrequencyBandExtractor(sampling_rate, n_channels)
        
        self.band_encoders = nn.ModuleDict()
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            self.band_encoders[band] = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=7, stride=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2),
                nn.Conv1d(32, 64, kernel_size=5, stride=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2),
                nn.Conv1d(64, 128, kernel_size=3, stride=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 * 5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.output_size = 512
    
    def forward(self, x):
        """Scipy 기반 순전파"""
        band_features = self.frequency_extractor(x)
        encoded_bands = {}
        band_vectors = []
        
        for band, feature in band_features.items():
            encoded = self.band_encoders[band](feature)
            pooled = self.global_pool(encoded).squeeze(-1)
            encoded_bands[band] = encoded
            band_vectors.append(pooled)
        
        fused_features = torch.cat(band_vectors, dim=1)
        final_features = self.feature_fusion(fused_features)
        return final_features, encoded_bands, band_features


class ScipyMultiHead_SimCLR_Model(nn.Module):
    """Scipy 기반 다중 헤드 SimCLR 모델"""
    def __init__(self, encoder, projection_head):
        super(ScipyMultiHead_SimCLR_Model, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, x, compute_temperature=False, other_view=None):
        """Scipy 기반 순전파"""
        encoded_features, encoded_bands, raw_band_features = self.encoder(x)
        
        if compute_temperature and other_view is not None:
            with torch.no_grad():
                _, other_encoded_bands, _ = self.encoder(other_view)
                other_projections = self.projection_head(other_encoded_bands)
            
            projections, adaptive_temps = self.projection_head(
                encoded_bands,
                compute_temperature=True,
                view_pair=(self.projection_head(encoded_bands), other_projections)
            )
            return encoded_features, projections, adaptive_temps
        else:
            projections = self.projection_head(encoded_bands)
            return encoded_features, projections


def create_scipy_multihead_simclr_model(n_channels=N_CHANNELS, sampling_rate=500):
    """Scipy 기반 다중 헤드 SimCLR 모델 생성"""
    encoder = ScipyMultiFrequency_SimCLR_Encoder(n_channels=n_channels, sampling_rate=sampling_rate)
    projection_head = OptimizedMultiFrequencyProjectionHead(
        input_dim=128,
        hidden_dim=256,
        output_dim=128,
        temp_min=0.05,
        temp_max=0.5
    )
    model = ScipyMultiHead_SimCLR_Model(encoder, projection_head)
    return model


class SingleHead_SimCLR_Model(nn.Module):
    """Single-head baseline for the cost comparison: one projection head on the fused 512-d feature,
    fixed temperature (no temperature networks)."""
    def __init__(self, encoder, hidden_dim=SIMCLR_HIDDEN_DIM, output_dim=SIMCLR_OUTPUT_DIM):
        super(SingleHead_SimCLR_Model, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.output_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        encoded_features, _, _ = self.encoder(x)
        return encoded_features, F.normalize(self.projection_head(encoded_features), p=2, dim=1)


def build_linear_head(input_dim=512, n_classes=2):
    """Linear-evaluation head: a single affine layer on frozen encoder features"""
    return nn.Linear(input_dim, n_classes)


class ScipyFineTuningClassifier(nn.Module):
    """Encoder + MLP classifier, all parameters updated (fine-tuning protocol)"""
    def __init__(self, pretrained_encoder_state, encoder_config, n_classes=2, freeze_encoder=False):
        super(ScipyFineTuningClassifier, self).__init__()
        self.encoder = ScipyMultiFrequency_SimCLR_Encoder(**encoder_config)
        self.encoder.load_state_dict(pretrained_encoder_state)
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_size, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            # A frozen encoder must also keep its BatchNorm running statistics fixed
            self.encoder.eval()
        return self

    def forward(self, raw_data):
        """Scipy 기반 순전파"""
        encoded_features, _, _ = self.encoder(raw_data)
        output = self.classifier(encoded_features)
        return output

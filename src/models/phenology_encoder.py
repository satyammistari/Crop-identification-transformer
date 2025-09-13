"""
PhenologyEncoder: Temporal encoder for crop phenological stage inference.

This module implements a deep temporal encoder that processes multi-temporal 
satellite imagery to infer crop phenological stages (sowing, vegetative, 
flowering, maturity) which are then used for dynamic cross-modal attention.

Key components:
- Global spatial pooling to reduce spatial dimension
- 1D convolutions for temporal feature extraction
- Bidirectional LSTM for temporal modeling
- Multi-head attention for temporal weighting
- Classification head for phenological stage prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PhenologyEncoder(nn.Module):
    """
    Encode temporal satellite data to infer crop phenological stages.
    
    This encoder processes time series satellite data to predict crop growth stages
    and extract temporal features for downstream fusion.
    
    Args:
        input_channels (int): Number of satellite bands (e.g., 6 for optical)
        temporal_length (int): Number of time steps (e.g., 6 months)
        hidden_dim (int): Hidden dimension for LSTM and attention
        num_phenology_classes (int): Number of phenological stages (default: 4)
        conv_channels (list): Channel dimensions for temporal convolutions
        lstm_layers (int): Number of LSTM layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        temporal_length: int = 6,
        hidden_dim: int = 128,
        num_phenology_classes: int = 4,
        conv_channels: Optional[list] = None,
        lstm_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super(PhenologyEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.temporal_length = temporal_length
        self.hidden_dim = hidden_dim
        self.num_phenology_classes = num_phenology_classes
        self.num_heads = num_heads
        
        # Default convolution channel progression
        if conv_channels is None:
            conv_channels = [64, 128, hidden_dim]
        
        # Global spatial pooling - reduce [B, T, C, H, W] to [B, T, C]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal feature extraction with 1D convolutions
        self.temporal_convs = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_channels:
            self.temporal_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout1d(dropout)
            ))
            in_channels = out_channels
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles the output size
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Multi-head temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head for phenological stages
        self.phenology_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_phenology_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1 for better gradient flow
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.)
    
    def forward(
        self, 
        time_series: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the phenology encoder.
        
        Args:
            time_series (torch.Tensor): Input tensor [B, T, C, H, W]
                - B: batch size
                - T: temporal length (e.g., 6 months)
                - C: number of bands (e.g., 6 for optical)
                - H, W: spatial dimensions
            temporal_mask (torch.Tensor, optional): Mask for variable length sequences [B, T]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - phenology_probs: Phenological stage probabilities [B, num_classes]
                - temporal_features: Temporal features for each time step [B, T, hidden_dim]
        """
        batch_size, temporal_length, channels, height, width = time_series.shape
        
        # Global spatial pooling: [B, T, C, H, W] -> [B, T, C]
        pooled_features = self.global_pool(time_series.view(-1, channels, height, width))
        pooled_features = pooled_features.view(batch_size, temporal_length, channels)
        
        # Transpose for 1D convolution: [B, T, C] -> [B, C, T]
        conv_input = pooled_features.transpose(1, 2)
        
        # Apply temporal convolutions
        conv_features = conv_input
        for conv_layer in self.temporal_convs:
            conv_features = conv_layer(conv_features)
        
        # Transpose back: [B, C, T] -> [B, T, C]
        conv_features = conv_features.transpose(1, 2)
        
        # LSTM for temporal modeling
        if temporal_mask is not None:
            # Pack padded sequences for efficiency with variable lengths
            lengths = temporal_mask.sum(dim=1).cpu()
            packed_input = nn.utils.rnn.pack_padded_sequence(
                conv_features, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_input)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(conv_features)
        
        # Multi-head temporal attention
        # Self-attention over temporal dimension
        attention_mask = None
        if temporal_mask is not None:
            # Create attention mask: True for positions to mask
            attention_mask = ~temporal_mask.bool()
        
        attended_features, attention_weights = self.temporal_attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=attention_mask
        )
        
        # Residual connection and layer norm
        temporal_features = self.layer_norm(attended_features + lstm_output)
        
        # Global temporal pooling for classification
        if temporal_mask is not None:
            # Masked average pooling
            temporal_mask_expanded = temporal_mask.unsqueeze(-1).expand_as(temporal_features)
            masked_features = temporal_features * temporal_mask_expanded
            global_features = masked_features.sum(dim=1) / temporal_mask_expanded.sum(dim=1)
        else:
            # Simple average pooling
            global_features = temporal_features.mean(dim=1)
        
        # Phenological stage classification
        phenology_logits = self.phenology_classifier(global_features)
        phenology_probs = F.softmax(phenology_logits, dim=-1)
        
        return phenology_probs, temporal_features
    
    def get_attention_weights(
        self, 
        time_series: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract attention weights for visualization.
        
        Args:
            time_series (torch.Tensor): Input tensor [B, T, C, H, W]
            temporal_mask (torch.Tensor, optional): Temporal mask [B, T]
        
        Returns:
            torch.Tensor: Attention weights [B, num_heads, T, T]
        """
        batch_size, temporal_length, channels, height, width = time_series.shape
        
        # Forward pass up to attention
        pooled_features = self.global_pool(time_series.view(-1, channels, height, width))
        pooled_features = pooled_features.view(batch_size, temporal_length, channels)
        conv_input = pooled_features.transpose(1, 2)
        
        conv_features = conv_input
        for conv_layer in self.temporal_convs:
            conv_features = conv_layer(conv_features)
        conv_features = conv_features.transpose(1, 2)
        
        lstm_output, _ = self.lstm(conv_features)
        
        # Get attention weights
        attention_mask = None
        if temporal_mask is not None:
            attention_mask = ~temporal_mask.bool()
        
        _, attention_weights = self.temporal_attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=attention_mask,
            average_attn_weights=False  # Return per-head weights
        )
        
        return attention_weights


class PositionalEncoding(nn.Module):
    """
    Add positional encoding to temporal features.
    
    This helps the model understand the temporal order and seasonal patterns.
    """
    
    def __init__(self, hidden_dim: int, max_length: int = 24):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x (torch.Tensor): Input tensor [B, T, hidden_dim]
        
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class EnhancedPhenologyEncoder(PhenologyEncoder):
    """
    Enhanced version with positional encoding and additional features.
    """
    
    def __init__(self, *args, use_positional_encoding: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(self.hidden_dim)
        else:
            self.pos_encoding = None
    
    def forward(
        self, 
        time_series: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward with positional encoding."""
        
        # Get features from parent class
        phenology_probs, temporal_features = super().forward(time_series, temporal_mask)
        
        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            temporal_features = self.pos_encoding(temporal_features)
        
        return phenology_probs, temporal_features


# Export for easy imports
__all__ = ['PhenologyEncoder', 'EnhancedPhenologyEncoder', 'PositionalEncoding']

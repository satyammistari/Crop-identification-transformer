"""
CrossModalPhenologicalAttention: Core innovation for dynamic cross-modal fusion.

This module implements the key innovation of the AMPT model: dynamic weighting
of SAR vs optical features based on crop phenological stages. This is the first
model to adapt cross-modal attention using real-time crop growth stages.

Key insight: Different crop growth stages require different sensor emphasis:
- Early (planting): SAR better for soil/moisture detection
- Mid (growth): Balanced SAR + optical  
- Late (harvest): Optical better for chlorophyll/maturity

Components:
- Phenological gating network for dynamic weights
- Cross-modal attention mechanisms
- Residual connections and normalization
- Attention visualization for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class CrossModalPhenologicalAttention(nn.Module):
    """
    Dynamic cross-modal attention that adapts based on crop phenological stages.
    
    This module implements the core innovation: using phenological stage information
    to dynamically weight SAR vs optical features for optimal crop classification.
    
    Args:
        optical_dim (int): Dimension of optical features
        sar_dim (int): Dimension of SAR features
        common_dim (int): Common projection dimension
        num_heads (int): Number of attention heads
        phenology_classes (int): Number of phenological stages
        dropout (float): Dropout probability
        temperature (float): Temperature for attention scaling
        use_residual (bool): Whether to use residual connections
    """
    
    def __init__(
        self,
        optical_dim: int = 256,
        sar_dim: int = 128,
        common_dim: int = 256,
        num_heads: int = 8,
        phenology_classes: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.1,
        use_residual: bool = True
    ):
        super(CrossModalPhenologicalAttention, self).__init__()
        
        self.optical_dim = optical_dim
        self.sar_dim = sar_dim
        self.common_dim = common_dim
        self.num_heads = num_heads
        self.phenology_classes = phenology_classes
        self.temperature = temperature
        self.use_residual = use_residual
        
        assert common_dim % num_heads == 0, "common_dim must be divisible by num_heads"
        self.head_dim = common_dim // num_heads
        
        # Phenological gating network
        # Maps phenology stage probabilities to modality weights
        self.phenology_gate = nn.Sequential(
            nn.Linear(phenology_classes, common_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(common_dim, 2),  # 2 weights: SAR and optical
            nn.Softmax(dim=-1)
        )
        
        # Project features to common dimension
        self.optical_projection = nn.Linear(optical_dim, common_dim)
        self.sar_projection = nn.Linear(sar_dim, common_dim)
        
        # Self-attention within modalities
        self.optical_self_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.sar_self_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal attention (SAR queries optical)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.optical_norm1 = nn.LayerNorm(common_dim)
        self.optical_norm2 = nn.LayerNorm(common_dim)
        self.sar_norm1 = nn.LayerNorm(common_dim)
        self.sar_norm2 = nn.LayerNorm(common_dim)
        self.cross_norm = nn.LayerNorm(common_dim)
        
        # Feed-forward networks
        self.optical_ffn = self._build_ffn(common_dim, dropout)
        self.sar_ffn = self._build_ffn(common_dim, dropout)
        self.cross_ffn = self._build_ffn(common_dim, dropout)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_ffn(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Build feed-forward network."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        optical_features: torch.Tensor,
        sar_features: torch.Tensor,
        phenology_stage: torch.Tensor,
        optical_mask: Optional[torch.Tensor] = None,
        sar_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through cross-modal phenological attention.
        
        Args:
            optical_features (torch.Tensor): Optical features [B, N_opt, optical_dim]
            sar_features (torch.Tensor): SAR features [B, N_sar, sar_dim]
            phenology_stage (torch.Tensor): Phenology probabilities [B, phenology_classes]
            optical_mask (torch.Tensor, optional): Mask for optical features [B, N_opt]
            sar_mask (torch.Tensor, optional): Mask for SAR features [B, N_sar]
        
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - fused_features: Fused multi-modal features [B, N, common_dim]
                - attention_info: Dictionary with attention weights and modality weights
        """
        batch_size = optical_features.size(0)
        
        # Generate phenology-based modality weights
        modality_weights = self.phenology_gate(phenology_stage)  # [B, 2]
        sar_weight = modality_weights[:, 0:1]  # [B, 1]
        optical_weight = modality_weights[:, 1:2]  # [B, 1]
        
        # Project to common dimension
        optical_proj = self.optical_projection(optical_features)  # [B, N_opt, common_dim]
        sar_proj = self.sar_projection(sar_features)  # [B, N_sar, common_dim]
        
        # Apply phenological weights
        optical_weighted = optical_proj * optical_weight.unsqueeze(1)
        sar_weighted = sar_proj * sar_weight.unsqueeze(1)
        
        # Self-attention within each modality
        optical_self_attn, optical_self_weights = self.optical_self_attention(
            query=optical_weighted,
            key=optical_weighted,
            value=optical_weighted,
            key_padding_mask=~optical_mask.bool() if optical_mask is not None else None
        )
        
        sar_self_attn, sar_self_weights = self.sar_self_attention(
            query=sar_weighted,
            key=sar_weighted,
            value=sar_weighted,
            key_padding_mask=~sar_mask.bool() if sar_mask is not None else None
        )
        
        # Residual connections and normalization
        if self.use_residual:
            optical_self_attn = self.optical_norm1(optical_self_attn + optical_weighted)
            sar_self_attn = self.sar_norm1(sar_self_attn + sar_weighted)
        else:
            optical_self_attn = self.optical_norm1(optical_self_attn)
            sar_self_attn = self.sar_norm1(sar_self_attn)
        
        # Feed-forward networks
        optical_ffn_out = self.optical_ffn(optical_self_attn)
        sar_ffn_out = self.sar_ffn(sar_self_attn)
        
        if self.use_residual:
            optical_features_refined = self.optical_norm2(optical_ffn_out + optical_self_attn)
            sar_features_refined = self.sar_norm2(sar_ffn_out + sar_self_attn)
        else:
            optical_features_refined = self.optical_norm2(optical_ffn_out)
            sar_features_refined = self.sar_norm2(sar_ffn_out)
        
        # Cross-modal attention: SAR queries optical
        cross_attn_out, cross_attn_weights = self.cross_attention(
            query=sar_features_refined,
            key=optical_features_refined,
            value=optical_features_refined,
            key_padding_mask=~optical_mask.bool() if optical_mask is not None else None
        )
        
        # Normalize cross-attention output
        cross_attn_out = self.cross_norm(cross_attn_out)
        
        # Cross-modal feed-forward
        cross_ffn_out = self.cross_ffn(cross_attn_out)
        
        # Final fusion: concatenate SAR and enhanced cross-modal features
        # We align dimensions by taking the first N_sar features from optical if needed
        n_sar = sar_features_refined.size(1)
        if optical_features_refined.size(1) >= n_sar:
            optical_aligned = optical_features_refined[:, :n_sar, :]
        else:
            # Repeat or interpolate if optical has fewer features
            optical_aligned = F.interpolate(
                optical_features_refined.transpose(1, 2),
                size=n_sar,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Concatenate for fusion
        concat_features = torch.cat([sar_features_refined, cross_ffn_out], dim=-1)
        fused_features = self.fusion_layer(concat_features)
        
        # Prepare attention information for visualization and analysis
        attention_info = {
            'modality_weights': modality_weights,
            'sar_weight': sar_weight,
            'optical_weight': optical_weight,
            'optical_self_weights': optical_self_weights,
            'sar_self_weights': sar_self_weights,
            'cross_attention_weights': cross_attn_weights,
            'phenology_stage': phenology_stage
        }
        
        return fused_features, attention_info
    
    def get_phenological_insights(
        self,
        phenology_stage: torch.Tensor
    ) -> Dict[str, str]:
        """
        Get interpretable insights about phenological stage effects.
        
        Args:
            phenology_stage (torch.Tensor): Phenology probabilities [B, phenology_classes]
        
        Returns:
            Dict[str, str]: Human-readable insights about the phenological weighting
        """
        # Stage names
        stage_names = ['sowing', 'vegetative', 'flowering', 'maturity']
        
        # Get modality weights
        modality_weights = self.phenology_gate(phenology_stage)
        sar_weight = modality_weights[:, 0].item()
        optical_weight = modality_weights[:, 1].item()
        
        # Determine dominant stage
        dominant_stage_idx = torch.argmax(phenology_stage, dim=-1).item()
        dominant_stage = stage_names[dominant_stage_idx]
        confidence = phenology_stage[0, dominant_stage_idx].item()
        
        # Generate insights
        insights = {
            'dominant_stage': dominant_stage,
            'stage_confidence': f"{confidence:.3f}",
            'sar_weight': f"{sar_weight:.3f}",
            'optical_weight': f"{optical_weight:.3f}",
            'interpretation': self._get_stage_interpretation(dominant_stage, sar_weight, optical_weight)
        }
        
        return insights
    
    def _get_stage_interpretation(self, stage: str, sar_weight: float, optical_weight: float) -> str:
        """Generate human-readable interpretation of the weighting."""
        
        if stage == 'sowing':
            if sar_weight > optical_weight:
                return "Early season: SAR is emphasized for soil moisture and texture detection."
            else:
                return "Early season: Optical dominance suggests visible crop emergence."
        
        elif stage == 'vegetative':
            if abs(sar_weight - optical_weight) < 0.1:
                return "Growth phase: Balanced SAR-optical fusion for vegetation monitoring."
            elif sar_weight > optical_weight:
                return "Growth phase: SAR emphasis for structural vegetation analysis."
            else:
                return "Growth phase: Optical emphasis for chlorophyll and vigor assessment."
        
        elif stage == 'flowering':
            if optical_weight > sar_weight:
                return "Flowering: Optical dominance for spectral changes and bloom detection."
            else:
                return "Flowering: Unexpected SAR emphasis - may indicate stress conditions."
        
        elif stage == 'maturity':
            if optical_weight > sar_weight:
                return "Maturity: Optical emphasis for senescence and harvest readiness."
            else:
                return "Maturity: SAR contribution suggests structural changes in crops."
        
        else:
            return f"Stage {stage}: SAR={sar_weight:.2f}, Optical={optical_weight:.2f}"


class PhenologyGuidedFusion(nn.Module):
    """
    Alternative simpler fusion approach based on phenological stages.
    """
    
    def __init__(
        self,
        optical_dim: int,
        sar_dim: int,
        phenology_classes: int = 4,
        output_dim: int = 256
    ):
        super(PhenologyGuidedFusion, self).__init__()
        
        # Stage-specific fusion weights
        self.stage_fusion_weights = nn.Parameter(
            torch.randn(phenology_classes, 2)  # [stages, modalities]
        )
        
        # Projection layers
        self.optical_proj = nn.Linear(optical_dim, output_dim)
        self.sar_proj = nn.Linear(sar_dim, output_dim)
        
        # Initialize stage-specific weights based on domain knowledge
        with torch.no_grad():
            # Early stage: favor SAR
            self.stage_fusion_weights[0] = torch.tensor([0.7, 0.3])  # [SAR, optical]
            # Growth stage: balanced
            self.stage_fusion_weights[1] = torch.tensor([0.5, 0.5])
            # Flowering: favor optical
            self.stage_fusion_weights[2] = torch.tensor([0.3, 0.7])
            # Maturity: favor optical
            self.stage_fusion_weights[3] = torch.tensor([0.2, 0.8])
    
    def forward(
        self,
        optical_features: torch.Tensor,
        sar_features: torch.Tensor,
        phenology_stage: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple phenology-guided fusion.
        
        Args:
            optical_features: [B, N, optical_dim]
            sar_features: [B, N, sar_dim]  
            phenology_stage: [B, phenology_classes]
        
        Returns:
            torch.Tensor: Fused features [B, N, output_dim]
        """
        # Project features
        optical_proj = self.optical_proj(optical_features)
        sar_proj = self.sar_proj(sar_features)
        
        # Compute stage-weighted fusion weights
        fusion_weights = torch.matmul(phenology_stage, self.stage_fusion_weights)  # [B, 2]
        sar_weight = fusion_weights[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
        optical_weight = fusion_weights[:, 1:2].unsqueeze(-1)  # [B, 1, 1]
        
        # Weighted fusion
        fused = sar_weight * sar_proj + optical_weight * optical_proj
        
        return fused


# Export for easy imports
__all__ = [
    'CrossModalPhenologicalAttention',
    'PhenologyGuidedFusion'
]

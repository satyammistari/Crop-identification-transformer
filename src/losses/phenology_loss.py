"""
PhenologyLoss: Loss for phenological consistency and weather alignment.

This module implements loss functions to ensure phenological consistency
across temporal predictions and alignment with weather patterns. This is
crucial for the temporal modeling aspect of the AMPT model.

Components:
- Classification loss for phenological stage prediction
- Temporal consistency loss to penalize abrupt changes
- Weather alignment loss to ensure predictions match weather patterns
- Growth rate consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class PhenologyLoss(nn.Module):
    """
    Loss function for phenological stage prediction and consistency.
    
    This loss ensures that:
    1. Phenological stages are correctly classified
    2. Temporal transitions are smooth and realistic
    3. Predictions align with weather conditions
    4. Growth rates are consistent with phenological stages
    
    Args:
        classification_weight (float): Weight for stage classification loss
        temporal_weight (float): Weight for temporal consistency loss
        weather_weight (float): Weight for weather alignment loss
        growth_weight (float): Weight for growth rate consistency loss
        temperature (float): Temperature for temporal smoothing
        ignore_index (int): Index to ignore in loss computation
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        temporal_weight: float = 0.5,
        weather_weight: float = 0.3,
        growth_weight: float = 0.2,
        temperature: float = 0.1,
        ignore_index: int = -100
    ):
        super(PhenologyLoss, self).__init__()
        
        self.classification_weight = classification_weight
        self.temporal_weight = temporal_weight
        self.weather_weight = weather_weight
        self.growth_weight = growth_weight
        self.temperature = temperature
        self.ignore_index = ignore_index
        
        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # MSE loss for regression tasks
        self.mse_loss = nn.MSELoss()
        
        # Smooth L1 loss for growth rate
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # Define expected phenology transitions
        self.register_buffer('transition_matrix', self._create_transition_matrix())
        
        # Define weather-phenology alignment patterns
        self.register_buffer('weather_patterns', self._create_weather_patterns())
    
    def _create_transition_matrix(self) -> torch.Tensor:
        """
        Create expected transition probabilities between phenological stages.
        
        Stages: 0=sowing, 1=vegetative, 2=flowering, 3=maturity
        
        Returns:
            torch.Tensor: Transition matrix [4, 4]
        """
        # Define realistic transition probabilities
        transitions = torch.tensor([
            [0.7, 0.3, 0.0, 0.0],  # From sowing: mostly stay, some to vegetative
            [0.0, 0.6, 0.4, 0.0],  # From vegetative: stay or to flowering
            [0.0, 0.0, 0.5, 0.5],  # From flowering: stay or to maturity
            [0.0, 0.0, 0.1, 0.9],  # From maturity: mostly stay, some back to flowering
        ])
        
        return transitions
    
    def _create_weather_patterns(self) -> torch.Tensor:
        """
        Create expected weather patterns for each phenological stage.
        
        Weather dimensions: [temperature, humidity, rainfall, solar, wind]
        
        Returns:
            torch.Tensor: Weather patterns [4, 5]
        """
        # Define typical weather preferences for each stage
        # Values are normalized expectations
        patterns = torch.tensor([
            [0.4, 0.6, 0.8, 0.5, 0.4],  # Sowing: moderate temp, high humidity/rain
            [0.6, 0.5, 0.6, 0.7, 0.5],  # Vegetative: warm, moderate humidity, high solar
            [0.7, 0.4, 0.4, 0.8, 0.3],  # Flowering: warm, low humidity/rain, high solar
            [0.5, 0.3, 0.2, 0.6, 0.6],  # Maturity: moderate temp, low humidity/rain
        ])
        
        return patterns
    
    def forward(
        self,
        phenology_predictions: torch.Tensor,
        phenology_targets: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        weather_data: Optional[torch.Tensor] = None,
        growth_rate_pred: Optional[torch.Tensor] = None,
        growth_rate_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute phenological loss components.
        
        Args:
            phenology_predictions (torch.Tensor): Predicted phenology probs [B, 4]
            phenology_targets (torch.Tensor, optional): True phenology labels [B]
            temporal_features (torch.Tensor, optional): Temporal features [B, T, D]
            weather_data (torch.Tensor, optional): Weather data [B, T, 5]
            growth_rate_pred (torch.Tensor, optional): Predicted growth rate [B, 1]
            growth_rate_target (torch.Tensor, optional): True growth rate [B, 1]
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Classification loss
        if phenology_targets is not None:
            classification_loss = self.classification_loss(
                phenology_predictions, phenology_targets
            )
            losses['classification'] = classification_loss
            total_loss += self.classification_weight * classification_loss
        
        # 2. Temporal consistency loss
        if temporal_features is not None and temporal_features.size(1) > 1:
            temporal_loss = self._compute_temporal_consistency_loss(
                phenology_predictions, temporal_features
            )
            losses['temporal_consistency'] = temporal_loss
            total_loss += self.temporal_weight * temporal_loss
        
        # 3. Weather alignment loss
        if weather_data is not None:
            weather_loss = self._compute_weather_alignment_loss(
                phenology_predictions, weather_data
            )
            losses['weather_alignment'] = weather_loss
            total_loss += self.weather_weight * weather_loss
        
        # 4. Growth rate consistency loss
        if growth_rate_pred is not None and growth_rate_target is not None:
            growth_loss = self.smooth_l1_loss(growth_rate_pred, growth_rate_target)
            losses['growth_rate'] = growth_loss
            total_loss += self.growth_weight * growth_loss
        elif growth_rate_pred is not None:
            # Consistency with phenological stage even without targets
            stage_growth_loss = self._compute_stage_growth_consistency(
                phenology_predictions, growth_rate_pred
            )
            losses['stage_growth_consistency'] = stage_growth_loss
            total_loss += self.growth_weight * stage_growth_loss
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_temporal_consistency_loss(
        self,
        phenology_predictions: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Penalizes abrupt changes in phenological predictions across time.
        
        Args:
            phenology_predictions (torch.Tensor): Current phenology predictions [B, 4]
            temporal_features (torch.Tensor): Temporal features [B, T, D]
        
        Returns:
            torch.Tensor: Temporal consistency loss
        """
        batch_size, temporal_length, feature_dim = temporal_features.shape
        
        # Predict phenology for each temporal step
        # Simple linear layer for temporal phenology prediction
        if not hasattr(self, 'temporal_phenology_head'):
            self.temporal_phenology_head = nn.Linear(feature_dim, 4).to(temporal_features.device)
        
        temporal_phenology_logits = self.temporal_phenology_head(temporal_features)  # [B, T, 4]
        temporal_phenology_probs = F.softmax(temporal_phenology_logits, dim=-1)
        
        # Compute transition consistency loss
        transition_loss = 0.0
        
        for t in range(temporal_length - 1):
            current_probs = temporal_phenology_probs[:, t, :]  # [B, 4]
            next_probs = temporal_phenology_probs[:, t + 1, :]  # [B, 4]
            
            # Compute expected next stage probabilities based on transition matrix
            expected_next_probs = torch.matmul(current_probs, self.transition_matrix)  # [B, 4]
            
            # KL divergence between expected and actual next probabilities
            kl_loss = F.kl_div(
                F.log_softmax(next_probs / self.temperature, dim=-1),
                F.softmax(expected_next_probs / self.temperature, dim=-1),
                reduction='batchmean'
            )
            
            transition_loss += kl_loss
        
        # Normalize by number of transitions
        transition_loss = transition_loss / (temporal_length - 1)
        
        # Add smoothness penalty for abrupt changes
        diff_loss = 0.0
        for t in range(temporal_length - 1):
            prob_diff = temporal_phenology_probs[:, t + 1, :] - temporal_phenology_probs[:, t, :]
            diff_loss += torch.mean(torch.sum(prob_diff ** 2, dim=-1))
        
        diff_loss = diff_loss / (temporal_length - 1)
        
        return transition_loss + 0.1 * diff_loss
    
    def _compute_weather_alignment_loss(
        self,
        phenology_predictions: torch.Tensor,
        weather_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weather alignment loss.
        
        Ensures phenological predictions align with weather conditions.
        
        Args:
            phenology_predictions (torch.Tensor): Phenology predictions [B, 4]
            weather_data (torch.Tensor): Weather data [B, T, 5]
        
        Returns:
            torch.Tensor: Weather alignment loss
        """
        batch_size = phenology_predictions.size(0)
        
        # Average weather across temporal dimension
        avg_weather = weather_data.mean(dim=1)  # [B, 5]
        
        # Normalize weather data to [0, 1] range for comparison
        # Assuming weather data is already normalized with known stats
        normalized_weather = torch.sigmoid(avg_weather)  # [B, 5]
        
        # Compute expected weather pattern based on phenology predictions
        expected_weather = torch.matmul(phenology_predictions, self.weather_patterns)  # [B, 5]
        
        # Compute MSE between expected and actual weather patterns
        weather_alignment_loss = self.mse_loss(normalized_weather, expected_weather)
        
        return weather_alignment_loss
    
    def _compute_stage_growth_consistency(
        self,
        phenology_predictions: torch.Tensor,
        growth_rate_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency between phenological stage and growth rate.
        
        Args:
            phenology_predictions (torch.Tensor): Phenology predictions [B, 4]
            growth_rate_pred (torch.Tensor): Predicted growth rate [B, 1]
        
        Returns:
            torch.Tensor: Stage-growth consistency loss
        """
        # Define expected growth rates for each phenological stage
        # Higher growth rate during vegetative phase, lower during maturity
        expected_growth_rates = torch.tensor([0.2, 0.8, 0.6, 0.1], device=phenology_predictions.device)
        
        # Compute weighted expected growth rate
        expected_growth = torch.sum(phenology_predictions * expected_growth_rates.unsqueeze(0), dim=1, keepdim=True)
        
        # Compute MSE between predicted and expected growth rates
        growth_consistency_loss = self.mse_loss(growth_rate_pred, expected_growth)
        
        return growth_consistency_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Standalone temporal consistency loss.
    
    Ensures smooth transitions in temporal predictions.
    """
    
    def __init__(self, smoothness_weight: float = 1.0):
        super(TemporalConsistencyLoss, self).__init__()
        self.smoothness_weight = smoothness_weight
    
    def forward(self, temporal_predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            temporal_predictions (torch.Tensor): Temporal predictions [B, T, C]
        
        Returns:
            torch.Tensor: Temporal consistency loss
        """
        if temporal_predictions.size(1) <= 1:
            return torch.tensor(0.0, device=temporal_predictions.device)
        
        # Compute temporal differences
        temporal_diffs = temporal_predictions[:, 1:, :] - temporal_predictions[:, :-1, :]
        
        # L2 norm of temporal differences
        consistency_loss = torch.mean(torch.sum(temporal_diffs ** 2, dim=-1))
        
        return self.smoothness_weight * consistency_loss


class WeatherPhenologyAlignmentLoss(nn.Module):
    """
    Standalone weather-phenology alignment loss.
    
    Ensures phenological predictions are consistent with weather conditions.
    """
    
    def __init__(self, alignment_patterns: Optional[torch.Tensor] = None):
        super(WeatherPhenologyAlignmentLoss, self).__init__()
        
        if alignment_patterns is None:
            # Default weather-phenology alignment patterns
            alignment_patterns = torch.tensor([
                [0.4, 0.6, 0.8, 0.5, 0.4],  # Sowing
                [0.6, 0.5, 0.6, 0.7, 0.5],  # Vegetative
                [0.7, 0.4, 0.4, 0.8, 0.3],  # Flowering
                [0.5, 0.3, 0.2, 0.6, 0.6],  # Maturity
            ])
        
        self.register_buffer('alignment_patterns', alignment_patterns)
    
    def forward(
        self,
        phenology_probs: torch.Tensor,
        weather_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weather-phenology alignment loss.
        
        Args:
            phenology_probs (torch.Tensor): Phenology probabilities [B, 4]
            weather_data (torch.Tensor): Weather data [B, weather_dim]
        
        Returns:
            torch.Tensor: Alignment loss
        """
        # Compute expected weather pattern
        expected_weather = torch.matmul(phenology_probs, self.alignment_patterns)
        
        # Normalize actual weather to [0, 1]
        normalized_weather = torch.sigmoid(weather_data)
        
        # Compute MSE loss
        alignment_loss = F.mse_loss(normalized_weather, expected_weather)
        
        return alignment_loss


# Export for easy imports
__all__ = [
    'PhenologyLoss',
    'TemporalConsistencyLoss',
    'WeatherPhenologyAlignmentLoss'
]

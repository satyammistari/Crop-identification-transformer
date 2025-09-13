"""
CombinedLoss: Multi-objective loss combining all loss components.

This module implements the combined loss function that brings together:
1. Segmentation losses (CrossEntropy, Dice, Focal)
2. Phenological consistency losses
3. Growth rate prediction losses
4. Auxiliary task losses

The combined loss enables end-to-end training of the complete AMPT model
with proper weighting of different objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from .selective_loss import SelectiveLoss, FocalLoss, DiceLoss
from .phenology_loss import PhenologyLoss


class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-objective AMPT training.
    
    This loss function combines multiple objectives:
    - Segmentation: Main crop classification task
    - Phenology: Temporal phenological stage prediction
    - Growth: Growth rate regression
    - Consistency: Temporal and cross-modal consistency
    
    Args:
        num_classes (int): Number of crop classes
        segmentation_weight (float): Weight for segmentation loss
        phenology_weight (float): Weight for phenology loss
        growth_weight (float): Weight for growth rate loss
        consistency_weight (float): Weight for consistency losses
        class_weights (Optional[torch.Tensor]): Class weights for imbalanced data
        segmentation_loss_type (str): Type of segmentation loss
        use_auxiliary_losses (bool): Whether to include auxiliary losses
        ignore_index (int): Index to ignore in loss computation
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        segmentation_weight: float = 1.0,
        phenology_weight: float = 0.3,
        growth_weight: float = 0.2,
        consistency_weight: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        segmentation_loss_type: str = 'cross_entropy',
        use_auxiliary_losses: bool = True,
        ignore_index: int = 255,
        **kwargs
    ):
        super(CombinedLoss, self).__init__()
        
        self.num_classes = num_classes
        self.segmentation_weight = segmentation_weight
        self.phenology_weight = phenology_weight
        self.growth_weight = growth_weight
        self.consistency_weight = consistency_weight
        self.use_auxiliary_losses = use_auxiliary_losses
        self.ignore_index = ignore_index
        
        # Initialize segmentation loss
        # Filter out loss weight parameters that shouldn't go to SelectiveLoss
        selective_loss_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ['segmentation', 'phenology', 'growth', 'consistency']}
        
        self.segmentation_loss = SelectiveLoss(
            loss_type=segmentation_loss_type,
            class_weights=class_weights,
            ignore_index=ignore_index,
            **selective_loss_kwargs
        )
        
        # Initialize phenology loss
        if self.use_auxiliary_losses:
            self.phenology_loss = PhenologyLoss(
                classification_weight=1.0,
                temporal_weight=0.5,
                weather_weight=0.3,
                growth_weight=0.2,
                ignore_index=-100  # Different ignore index for phenology
            )
        
        # Initialize additional losses
        self.dice_loss = DiceLoss(smooth=1.0)
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # Loss balancing parameters
        self.adaptive_weighting = kwargs.get('adaptive_weighting', False)
        if self.adaptive_weighting:
            self.register_buffer('loss_weights_log', torch.zeros(4))  # For 4 main loss types
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss from model predictions.
        
        Args:
            predictions (Dict[str, torch.Tensor]): Model predictions containing:
                - 'segmentation': Segmentation logits [B, C, H, W]
                - 'phenology': Phenology probabilities [B, 4]
                - 'growth_rate': Growth rate predictions [B, 1]
                - 'attention_info': Attention information (optional)
            targets (torch.Tensor, optional): Ground truth labels [B, H, W]
            valid_mask (torch.Tensor, optional): Valid pixel mask [B, H, W]
            **kwargs: Additional targets and data
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Segmentation Loss (Primary objective)
        if 'segmentation' in predictions and targets is not None:
            seg_loss = self.segmentation_loss(
                predictions['segmentation'],
                targets,
                valid_mask
            )
            losses['segmentation_loss'] = seg_loss
            
            # Add Dice loss for better boundary handling
            dice_loss = self.dice_loss(predictions['segmentation'], targets)
            losses['dice_loss'] = dice_loss
            
            # Combined segmentation loss
            combined_seg_loss = 0.7 * seg_loss + 0.3 * dice_loss
            total_loss += self.segmentation_weight * combined_seg_loss
        
        # 2. Phenology Loss (Auxiliary objective)
        if self.use_auxiliary_losses and 'phenology' in predictions:
            phenology_targets = kwargs.get('phenology_targets')
            temporal_features = kwargs.get('temporal_features')
            weather_data = kwargs.get('weather_data')
            growth_rate_target = kwargs.get('growth_rate_target')
            
            phenology_losses = self.phenology_loss(
                phenology_predictions=predictions['phenology'],
                phenology_targets=phenology_targets,
                temporal_features=temporal_features,
                weather_data=weather_data,
                growth_rate_pred=predictions.get('growth_rate'),
                growth_rate_target=growth_rate_target
            )
            
            # Add individual phenology loss components
            for loss_name, loss_value in phenology_losses.items():
                if loss_name != 'total':
                    losses[f'phenology_{loss_name}'] = loss_value
            
            total_loss += self.phenology_weight * phenology_losses['total']
        
        # 3. Growth Rate Loss
        if 'growth_rate' in predictions:
            growth_target = kwargs.get('growth_rate_target')
            if growth_target is not None:
                growth_loss = self.smooth_l1_loss(predictions['growth_rate'], growth_target)
                losses['growth_rate_loss'] = growth_loss
                total_loss += self.growth_weight * growth_loss
        
        # 4. Consistency Losses
        if self.use_auxiliary_losses and 'attention_info' in predictions:
            consistency_loss = self._compute_consistency_losses(
                predictions['attention_info'],
                predictions.get('phenology')
            )
            losses.update(consistency_loss)
            
            # Add consistency loss to total
            if 'attention_consistency' in consistency_loss:
                total_loss += self.consistency_weight * consistency_loss['attention_consistency']
        
        # 5. Adaptive Loss Weighting (optional)
        if self.adaptive_weighting:
            total_loss = self._apply_adaptive_weighting(losses, total_loss)
        
        # Add total loss
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_consistency_losses(
        self,
        attention_info: Dict[str, torch.Tensor],
        phenology_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute consistency losses from attention information.
        
        Args:
            attention_info (Dict[str, torch.Tensor]): Attention weights and info
            phenology_probs (torch.Tensor, optional): Phenology probabilities
        
        Returns:
            Dict[str, torch.Tensor]: Consistency loss components
        """
        consistency_losses = {}
        
        # Attention consistency: ensure attention weights sum to 1 and are reasonable
        if 'modality_weights' in attention_info:
            modality_weights = attention_info['modality_weights']  # [B, 2]
            
            # Ensure weights are properly normalized (they should sum to 1)
            weight_sum = modality_weights.sum(dim=1, keepdim=True)
            weight_norm_loss = F.mse_loss(weight_sum, torch.ones_like(weight_sum))
            
            # Encourage diversity in modality usage (avoid collapse to single modality)
            weight_entropy = -torch.sum(modality_weights * torch.log(modality_weights + 1e-8), dim=1)
            weight_diversity_loss = F.mse_loss(weight_entropy, torch.ones_like(weight_entropy) * 0.693)  # log(2)
            
            attention_consistency = weight_norm_loss + 0.1 * weight_diversity_loss
            consistency_losses['attention_consistency'] = attention_consistency
        
        # Phenology-attention alignment: attention should align with phenology
        if phenology_probs is not None and 'sar_weight' in attention_info:
            sar_weight = attention_info['sar_weight'].squeeze(-1)  # [B]
            
            # Expected SAR weight based on phenology (higher for early stages)
            # Stages: 0=sowing, 1=vegetative, 2=flowering, 3=maturity
            expected_sar_weights = torch.tensor([0.7, 0.5, 0.3, 0.2], device=phenology_probs.device)
            expected_sar_weight = torch.sum(phenology_probs * expected_sar_weights.unsqueeze(0), dim=1)
            
            phenology_attention_loss = F.mse_loss(sar_weight, expected_sar_weight)
            consistency_losses['phenology_attention_alignment'] = phenology_attention_loss
        
        return consistency_losses
    
    def _apply_adaptive_weighting(
        self,
        losses: Dict[str, torch.Tensor],
        total_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adaptive loss weighting based on loss magnitudes.
        
        Args:
            losses (Dict[str, torch.Tensor]): Individual loss components
            total_loss (torch.Tensor): Current total loss
        
        Returns:
            torch.Tensor: Adaptively weighted total loss
        """
        # Extract main loss components
        main_losses = []
        loss_names = ['segmentation_loss', 'phenology_total', 'growth_rate_loss', 'attention_consistency']
        
        for loss_name in loss_names:
            if loss_name in losses:
                main_losses.append(losses[loss_name])
            else:
                main_losses.append(torch.tensor(0.0, device=total_loss.device))
        
        if len(main_losses) == 0:
            return total_loss
        
        # Convert to tensor
        main_losses = torch.stack(main_losses)
        
        # Compute adaptive weights using uncertainty weighting
        # log(σ²) is learned, where σ² is the task-dependent uncertainty
        weights = torch.exp(-self.loss_weights_log)
        regularization = self.loss_weights_log.sum()
        
        # Weighted combination
        weighted_losses = weights * main_losses
        adaptive_total = weighted_losses.sum() + regularization
        
        return adaptive_total
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get current loss weights for logging and analysis.
        
        Returns:
            Dict[str, float]: Current loss weights
        """
        weights = {
            'segmentation': self.segmentation_weight,
            'phenology': self.phenology_weight,
            'growth': self.growth_weight,
            'consistency': self.consistency_weight
        }
        
        if self.adaptive_weighting:
            adaptive_weights = torch.exp(-self.loss_weights_log).detach().cpu().numpy()
            weights.update({
                'adaptive_seg': adaptive_weights[0],
                'adaptive_phen': adaptive_weights[1],
                'adaptive_growth': adaptive_weights[2],
                'adaptive_consistency': adaptive_weights[3]
            })
        
        return weights
    
    def update_loss_weights(self, **new_weights):
        """
        Update loss weights during training.
        
        Args:
            **new_weights: New weight values
        """
        for weight_name, weight_value in new_weights.items():
            if hasattr(self, f'{weight_name}_weight'):
                setattr(self, f'{weight_name}_weight', weight_value)


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for handling different spatial resolutions.
    
    Applies loss at multiple scales to handle crops of different sizes.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        scales: list = [1.0, 0.5, 0.25],
        scale_weights: Optional[list] = None
    ):
        super(MultiScaleLoss, self).__init__()
        
        self.base_loss = base_loss
        self.scales = scales
        
        if scale_weights is None:
            scale_weights = [1.0] * len(scales)
        self.scale_weights = scale_weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            predictions (torch.Tensor): Predictions [B, C, H, W]
            targets (torch.Tensor): Targets [B, H, W]
            valid_mask (torch.Tensor, optional): Valid mask [B, H, W]
        
        Returns:
            torch.Tensor: Multi-scale loss
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1.0:
                # Original scale
                scale_loss = self.base_loss(predictions, targets, valid_mask)
            else:
                # Downsample
                h, w = int(predictions.size(-2) * scale), int(predictions.size(-1) * scale)
                
                pred_scaled = F.interpolate(predictions, size=(h, w), mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(targets.unsqueeze(1).float(), size=(h, w), mode='nearest')
                target_scaled = target_scaled.squeeze(1).long()
                
                if valid_mask is not None:
                    mask_scaled = F.interpolate(valid_mask.unsqueeze(1).float(), size=(h, w), mode='nearest')
                    mask_scaled = mask_scaled.squeeze(1).bool()
                else:
                    mask_scaled = None
                
                scale_loss = self.base_loss(pred_scaled, target_scaled, mask_scaled)
            
            total_loss += weight * scale_loss
        
        return total_loss / sum(self.scale_weights)


# Export for easy imports
__all__ = ['CombinedLoss', 'MultiScaleLoss']

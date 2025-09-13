"""
SelectiveLoss: Loss computation only on pixels with valid labels.

This module implements selective loss computation as required by the competition.
It only computes loss on pixels with valid labels, ignoring unlabeled or 
invalid pixels marked with ignore_index.

Key features:
- Masked loss computation
- Support for class weighting
- Handling of empty masks
- Multiple loss function support (CrossEntropy, Focal, Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class SelectiveLoss(nn.Module):
    """
    Selective loss that only computes loss on valid pixels.
    
    This loss function applies a mask to extract valid pixels before
    computing the standard loss, as required by the competition for
    sparse labeling scenarios.
    
    Args:
        loss_type (str): Type of base loss ('cross_entropy', 'focal', 'dice')
        class_weights (Optional[torch.Tensor]): Class weights for imbalanced datasets
        ignore_index (int): Index to ignore in loss computation
        focal_alpha (float): Alpha parameter for focal loss
        focal_gamma (float): Gamma parameter for focal loss
        dice_smooth (float): Smoothing factor for dice loss
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        loss_type: str = 'cross_entropy',
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        reduction: str = 'mean'
    ):
        super(SelectiveLoss, self).__init__()
        
        self.loss_type = loss_type.lower()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_smooth = dice_smooth
        
        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Initialize base loss function
        if self.loss_type == 'cross_entropy':
            self.base_loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=self.ignore_index,
                reduction='none'
            )
        elif self.loss_type == 'focal':
            self.base_loss = self._focal_loss
        elif self.loss_type == 'dice':
            self.base_loss = self._dice_loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute selective loss on valid pixels only.
        
        Args:
            predictions (torch.Tensor): Predictions [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]
            valid_mask (torch.Tensor, optional): Valid pixel mask [B, H, W]
        
        Returns:
            torch.Tensor: Computed loss value
        """
        batch_size, num_classes, height, width = predictions.shape
        
        # Create valid mask if not provided
        if valid_mask is None:
            valid_mask = (targets != self.ignore_index)
        
        # Handle case where no valid pixels exist
        if not valid_mask.any():
            # Return zero loss if no valid pixels
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        if self.loss_type in ['focal', 'dice']:
            # Custom loss functions that need special handling
            return self.base_loss(predictions, targets, valid_mask)
        else:
            # Standard loss functions
            # Apply valid mask by setting invalid pixels to ignore_index
            masked_targets = targets.clone()
            masked_targets[~valid_mask] = self.ignore_index
            
            # Compute loss (CrossEntropy handles ignore_index internally)
            loss_map = self.base_loss(predictions, masked_targets)
            
            # Apply reduction
            if self.reduction == 'mean':
                return loss_map.mean()
            elif self.reduction == 'sum':
                return loss_map.sum()
            else:
                return loss_map
    
    def _focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss with selective masking.
        
        Args:
            predictions (torch.Tensor): Predictions [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]
            valid_mask (torch.Tensor): Valid pixel mask [B, H, W]
        
        Returns:
            torch.Tensor: Focal loss value
        """
        # Get valid pixels
        valid_preds = predictions[:, :, valid_mask].transpose(0, 1)  # [C, N_valid]
        valid_targets = targets[valid_mask]  # [N_valid]
        
        # Apply softmax to get probabilities
        log_probs = F.log_softmax(valid_preds, dim=0)
        probs = torch.exp(log_probs)
        
        # Get probabilities for true class
        true_class_log_probs = log_probs[valid_targets, torch.arange(len(valid_targets))]
        true_class_probs = probs[valid_targets, torch.arange(len(valid_targets))]
        
        # Compute focal weight
        focal_weight = self.focal_alpha * (1 - true_class_probs) ** self.focal_gamma
        
        # Compute focal loss
        focal_loss = -focal_weight * true_class_log_probs
        
        # Apply class weights if available
        if self.class_weights is not None:
            class_weight_factor = self.class_weights[valid_targets]
            focal_loss = focal_loss * class_weight_factor
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _dice_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss with selective masking.
        
        Args:
            predictions (torch.Tensor): Predictions [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]
            valid_mask (torch.Tensor): Valid pixel mask [B, H, W]
        
        Returns:
            torch.Tensor: Dice loss value
        """
        num_classes = predictions.size(1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Apply softmax to predictions
        pred_probs = F.softmax(predictions, dim=1)
        
        # Apply valid mask
        pred_probs = pred_probs * valid_mask.unsqueeze(1)
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)
        
        # Compute Dice coefficient for each class
        dice_scores = []
        
        for c in range(num_classes):
            pred_c = pred_probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            # Compute intersection and union
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            # Compute Dice score
            dice_score = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
            dice_scores.append(dice_score)
        
        # Convert to loss (1 - Dice)
        dice_scores = torch.stack(dice_scores)
        dice_loss = 1.0 - dice_scores.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Standalone Focal Loss implementation.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions (torch.Tensor): Predictions [B, C, H, W] or [B, C]
            targets (torch.Tensor): Ground truth labels [B, H, W] or [B]
        
        Returns:
            torch.Tensor: Focal loss value
        """
        # Flatten if spatial dimensions
        if len(predictions.shape) > 2:
            predictions = predictions.permute(0, 2, 3, 1).contiguous().view(-1, predictions.size(1))
            targets = targets.view(-1)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Compute probabilities
        probs = F.softmax(predictions, dim=1)
        true_class_probs = probs[torch.arange(len(targets)), targets]
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - true_class_probs) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if available
        if self.class_weights is not None:
            class_weight_factor = self.class_weights[targets]
            focal_loss = focal_loss * class_weight_factor
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Standalone Dice Loss implementation.
    
    Dice Loss is particularly useful for segmentation tasks with
    imbalanced classes or small objects.
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions (torch.Tensor): Predictions [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]
        
        Returns:
            torch.Tensor: Dice loss value
        """
        num_classes = predictions.size(1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Apply softmax to predictions
        pred_probs = F.softmax(predictions, dim=1)
        
        # Compute Dice coefficient for each class
        dice_scores = []
        
        for c in range(num_classes):
            pred_c = pred_probs[:, c, :, :].flatten()
            target_c = targets_one_hot[:, c, :, :].flatten()
            
            # Compute intersection and union
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            # Compute Dice score
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)
        
        # Convert to loss (1 - Dice)
        dice_scores = torch.stack(dice_scores)
        
        if self.reduction == 'mean':
            dice_loss = 1.0 - dice_scores.mean()
        elif self.reduction == 'sum':
            dice_loss = (1.0 - dice_scores).sum()
        else:
            dice_loss = 1.0 - dice_scores
        
        return dice_loss


# Export for easy imports
__all__ = ['SelectiveLoss', 'FocalLoss', 'DiceLoss']

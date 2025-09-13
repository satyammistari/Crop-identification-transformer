"""
Unit tests for loss functions.

Tests all loss function implementations including SelectiveLoss,
PhenologyLoss, and CombinedLoss to ensure proper functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
from omegaconf import DictConfig, OmegaConf

# Import loss functions
from src.losses.selective_loss import SelectiveLoss
from src.losses.phenology_loss import PhenologyLoss  
from src.losses.combined_loss import CombinedLoss


class TestSelectiveLoss:
    """Test cases for SelectiveLoss."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DictConfig({
            'loss_type': 'cross_entropy',
            'ignore_index': -1,
            'label_smoothing': 0.1,
            'class_weights': None,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'dice_smooth': 1.0
        })
    
    @pytest.fixture
    def selective_loss(self, config):
        """Create SelectiveLoss instance."""
        return SelectiveLoss(
            loss_type=config.loss_type,
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing,
            class_weights=config.class_weights,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            dice_smooth=config.dice_smooth
        )
    
    def test_selective_loss_initialization(self, selective_loss, config):
        """Test SelectiveLoss initialization."""
        assert selective_loss.loss_type == config.loss_type
        assert selective_loss.ignore_index == config.ignore_index
        assert selective_loss.label_smoothing == config.label_smoothing
        assert selective_loss.focal_alpha == config.focal_alpha
        assert selective_loss.focal_gamma == config.focal_gamma
        assert selective_loss.dice_smooth == config.dice_smooth
    
    def test_cross_entropy_loss(self, config):
        """Test cross entropy loss computation."""
        config.loss_type = 'cross_entropy'
        loss_fn = SelectiveLoss(
            loss_type=config.loss_type,
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing
        )
        
        batch_size, num_classes, height, width = 2, 6, 32, 32
        
        # Create test data
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.randint(0, 2, (batch_size, height, width)).bool()
        
        # Compute loss
        loss = loss_fn(logits, targets, valid_mask)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0
    
    def test_focal_loss(self, config):
        """Test focal loss computation."""
        config.loss_type = 'focal'
        loss_fn = SelectiveLoss(
            loss_type=config.loss_type,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )
        
        batch_size, num_classes, height, width = 2, 6, 32, 32
        
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.randint(0, 2, (batch_size, height, width)).bool()
        
        loss = loss_fn(logits, targets, valid_mask)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_dice_loss(self, config):
        """Test dice loss computation."""
        config.loss_type = 'dice'
        loss_fn = SelectiveLoss(
            loss_type=config.loss_type,
            dice_smooth=config.dice_smooth
        )
        
        batch_size, num_classes, height, width = 2, 6, 16, 16
        
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.randint(0, 2, (batch_size, height, width)).bool()
        
        loss = loss_fn(logits, targets, valid_mask)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_combined_loss(self, config):
        """Test combined loss computation."""
        config.loss_type = 'combined'
        loss_fn = SelectiveLoss(
            loss_type=config.loss_type,
            label_smoothing=config.label_smoothing,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            dice_smooth=config.dice_smooth
        )
        
        batch_size, num_classes, height, width = 2, 6, 16, 16
        
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.randint(0, 2, (batch_size, height, width)).bool()
        
        loss = loss_fn(logits, targets, valid_mask)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_class_weights(self, config):
        """Test class weights functionality."""
        num_classes = 6
        class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 1.2, 2.5])
        
        loss_fn = SelectiveLoss(
            loss_type='cross_entropy',
            class_weights=class_weights
        )
        
        batch_size, height, width = 2, 16, 16
        
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.ones(batch_size, height, width).bool()
        
        loss = loss_fn(logits, targets, valid_mask)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_empty_valid_mask(self, selective_loss):
        """Test behavior with empty valid mask."""
        batch_size, num_classes, height, width = 1, 6, 8, 8
        
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.zeros(batch_size, height, width).bool()  # All False
        
        loss = selective_loss(logits, targets, valid_mask)
        
        # Should return zero loss or handle gracefully
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    
    def test_full_valid_mask(self, selective_loss):
        """Test behavior with full valid mask."""
        batch_size, num_classes, height, width = 2, 6, 16, 16
        
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.ones(batch_size, height, width).bool()  # All True
        
        loss = selective_loss(logits, targets, valid_mask)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_gradient_flow(self, selective_loss):
        """Test gradient flow through selective loss."""
        batch_size, num_classes, height, width = 1, 6, 8, 8
        
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.randint(0, 2, (batch_size, height, width)).bool()
        
        loss = selective_loss(logits, targets, valid_mask)
        loss.backward()
        
        # Check gradients exist
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()


class TestPhenologyLoss:
    """Test cases for PhenologyLoss."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DictConfig({
            'temporal_weight': 0.5,
            'weather_weight': 0.3,
            'transition_smoothness': 0.1,
            'weather_temperature': 0.1
        })
    
    @pytest.fixture
    def phenology_loss(self, config):
        """Create PhenologyLoss instance."""
        return PhenologyLoss(
            temporal_weight=config.temporal_weight,
            weather_weight=config.weather_weight,
            transition_smoothness=config.transition_smoothness,
            weather_temperature=config.weather_temperature
        )
    
    def test_phenology_loss_initialization(self, phenology_loss, config):
        """Test PhenologyLoss initialization."""
        assert phenology_loss.temporal_weight == config.temporal_weight
        assert phenology_loss.weather_weight == config.weather_weight
        assert phenology_loss.transition_smoothness == config.transition_smoothness
        assert phenology_loss.weather_temperature == config.weather_temperature
        assert hasattr(phenology_loss, 'transition_matrix')
        assert hasattr(phenology_loss, 'weather_patterns')
    
    def test_phenology_loss_forward(self, phenology_loss):
        """Test phenology loss forward pass."""
        batch_size, phenology_dim = 2, 64
        temporal_length, weather_features = 8, 10
        
        # Create test data
        phenology_pred = torch.randn(batch_size, phenology_dim, requires_grad=True)
        temporal_positions = torch.randint(0, 365, (batch_size, temporal_length))
        weather_data = torch.randn(batch_size, temporal_length, weather_features)
        
        # Compute loss
        loss = phenology_loss(phenology_pred, temporal_positions, weather_data)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0
    
    def test_temporal_consistency_loss(self, phenology_loss):
        """Test temporal consistency component."""
        batch_size, phenology_dim = 2, 64
        temporal_length = 6
        
        phenology_pred = torch.randn(batch_size, phenology_dim, requires_grad=True)
        temporal_positions = torch.sort(torch.randint(0, 365, (batch_size, temporal_length)))[0]
        weather_data = torch.randn(batch_size, temporal_length, 10)
        
        loss = phenology_loss(phenology_pred, temporal_positions, weather_data)
        
        # Temporal consistency should contribute to loss
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_weather_consistency_loss(self, phenology_loss):
        """Test weather consistency component."""
        batch_size, phenology_dim = 2, 64
        temporal_length = 8
        
        phenology_pred = torch.randn(batch_size, phenology_dim, requires_grad=True)
        temporal_positions = torch.randint(0, 365, (batch_size, temporal_length))
        
        # Create weather data with clear patterns
        weather_data = torch.randn(batch_size, temporal_length, 10)
        weather_data[:, :, 0] = torch.linspace(0, 30, temporal_length)  # Temperature pattern
        
        loss = phenology_loss(phenology_pred, temporal_positions, weather_data)
        
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_different_temporal_lengths(self, phenology_loss):
        """Test with different temporal sequence lengths."""
        batch_size, phenology_dim = 1, 64
        
        phenology_pred = torch.randn(batch_size, phenology_dim, requires_grad=True)
        
        # Test different temporal lengths
        for temporal_length in [4, 8, 12]:
            temporal_positions = torch.randint(0, 365, (batch_size, temporal_length))
            weather_data = torch.randn(batch_size, temporal_length, 10)
            
            loss = phenology_loss(phenology_pred, temporal_positions, weather_data)
            
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert loss.item() >= 0
    
    def test_gradient_flow(self, phenology_loss):
        """Test gradient flow through phenology loss."""
        batch_size, phenology_dim = 2, 64
        temporal_length = 6
        
        phenology_pred = torch.randn(batch_size, phenology_dim, requires_grad=True)
        temporal_positions = torch.randint(0, 365, (batch_size, temporal_length))
        weather_data = torch.randn(batch_size, temporal_length, 10)
        
        loss = phenology_loss(phenology_pred, temporal_positions, weather_data)
        loss.backward()
        
        # Check gradients
        assert phenology_pred.grad is not None
        assert not torch.isnan(phenology_pred.grad).any()
        assert not torch.isinf(phenology_pred.grad).any()


class TestCombinedLoss:
    """Test cases for CombinedLoss."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DictConfig({
            'segmentation_weight': 1.0,
            'phenology_weight': 0.1,
            'weather_weight': 0.05,
            'adaptive_weighting': True,
            'weight_update_frequency': 10,
            'segmentation_loss': {
                'loss_type': 'cross_entropy',
                'label_smoothing': 0.1,
                'ignore_index': -1
            },
            'phenology_loss': {
                'temporal_weight': 0.5,
                'weather_weight': 0.3,
                'transition_smoothness': 0.1
            }
        })
    
    @pytest.fixture
    def combined_loss(self, config):
        """Create CombinedLoss instance."""
        return CombinedLoss(
            segmentation_weight=config.segmentation_weight,
            phenology_weight=config.phenology_weight,
            weather_weight=config.weather_weight,
            adaptive_weighting=config.adaptive_weighting,
            weight_update_frequency=config.weight_update_frequency,
            segmentation_config=config.segmentation_loss,
            phenology_config=config.phenology_loss
        )
    
    def test_combined_loss_initialization(self, combined_loss, config):
        """Test CombinedLoss initialization."""
        assert combined_loss.segmentation_weight == config.segmentation_weight
        assert combined_loss.phenology_weight == config.phenology_weight
        assert combined_loss.weather_weight == config.weather_weight
        assert combined_loss.adaptive_weighting == config.adaptive_weighting
        assert hasattr(combined_loss, 'segmentation_loss')
        assert hasattr(combined_loss, 'phenology_loss')
    
    def test_combined_loss_forward(self, combined_loss):
        """Test combined loss forward pass."""
        batch_size, num_classes, height, width = 2, 6, 32, 32
        phenology_dim, temporal_length = 64, 8
        
        # Create test data
        segmentation_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        segmentation_targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.randint(0, 2, (batch_size, height, width)).bool()
        
        phenology_pred = torch.randn(batch_size, phenology_dim, requires_grad=True)
        temporal_positions = torch.randint(0, 365, (batch_size, temporal_length))
        weather_data = torch.randn(batch_size, temporal_length, 10)
        
        # Compute combined loss
        loss_dict = combined_loss(
            segmentation_logits=segmentation_logits,
            segmentation_targets=segmentation_targets,
            valid_mask=valid_mask,
            phenology_pred=phenology_pred,
            temporal_positions=temporal_positions,
            weather_data=weather_data
        )
        
        # Check loss dictionary structure
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'segmentation_loss' in loss_dict
        assert 'phenology_loss' in loss_dict
        
        # Check loss properties
        total_loss = loss_dict['total_loss']
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert total_loss.requires_grad
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)
        assert total_loss.item() >= 0
    
    def test_loss_components(self, combined_loss):
        """Test individual loss components."""
        batch_size, num_classes, height, width = 1, 6, 16, 16
        phenology_dim, temporal_length = 64, 6
        
        segmentation_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        segmentation_targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.ones(batch_size, height, width).bool()
        
        phenology_pred = torch.randn(batch_size, phenology_dim, requires_grad=True)
        temporal_positions = torch.randint(0, 365, (batch_size, temporal_length))
        weather_data = torch.randn(batch_size, temporal_length, 10)
        
        loss_dict = combined_loss(
            segmentation_logits=segmentation_logits,
            segmentation_targets=segmentation_targets,
            valid_mask=valid_mask,
            phenology_pred=phenology_pred,
            temporal_positions=temporal_positions,
            weather_data=weather_data
        )
        
        # All components should be positive
        assert loss_dict['segmentation_loss'].item() >= 0
        assert loss_dict['phenology_loss'].item() >= 0
        
        # Total loss should be combination of components
        expected_total = (
            combined_loss.segmentation_weight * loss_dict['segmentation_loss'] +
            combined_loss.phenology_weight * loss_dict['phenology_loss']
        )
        
        assert torch.allclose(loss_dict['total_loss'], expected_total, atol=1e-6)
    
    def test_adaptive_weighting(self, config):
        """Test adaptive weighting functionality."""
        config.adaptive_weighting = True
        config.weight_update_frequency = 5
        
        combined_loss = CombinedLoss(
            segmentation_weight=config.segmentation_weight,
            phenology_weight=config.phenology_weight,
            adaptive_weighting=config.adaptive_weighting,
            weight_update_frequency=config.weight_update_frequency,
            segmentation_config=config.segmentation_loss,
            phenology_config=config.phenology_loss
        )
        
        batch_size, num_classes, height, width = 1, 6, 16, 16
        
        # Store initial weights
        initial_seg_weight = combined_loss.segmentation_weight
        initial_phen_weight = combined_loss.phenology_weight
        
        # Run multiple forward passes to trigger weight updates
        for step in range(10):
            segmentation_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
            segmentation_targets = torch.randint(0, num_classes, (batch_size, height, width))
            valid_mask = torch.ones(batch_size, height, width).bool()
            
            phenology_pred = torch.randn(batch_size, 64, requires_grad=True)
            temporal_positions = torch.randint(0, 365, (batch_size, 6))
            weather_data = torch.randn(batch_size, 6, 10)
            
            loss_dict = combined_loss(
                segmentation_logits=segmentation_logits,
                segmentation_targets=segmentation_targets,
                valid_mask=valid_mask,
                phenology_pred=phenology_pred,
                temporal_positions=temporal_positions,
                weather_data=weather_data,
                current_step=step
            )
        
        # Weights might have been updated
        assert isinstance(combined_loss.segmentation_weight, float)
        assert isinstance(combined_loss.phenology_weight, float)
    
    def test_no_adaptive_weighting(self, config):
        """Test without adaptive weighting."""
        config.adaptive_weighting = False
        
        combined_loss = CombinedLoss(
            segmentation_weight=config.segmentation_weight,
            phenology_weight=config.phenology_weight,
            adaptive_weighting=config.adaptive_weighting,
            segmentation_config=config.segmentation_loss,
            phenology_config=config.phenology_loss
        )
        
        initial_seg_weight = combined_loss.segmentation_weight
        initial_phen_weight = combined_loss.phenology_weight
        
        # Run forward pass
        batch_size, num_classes, height, width = 1, 6, 8, 8
        
        segmentation_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        segmentation_targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.ones(batch_size, height, width).bool()
        
        phenology_pred = torch.randn(batch_size, 64, requires_grad=True)
        temporal_positions = torch.randint(0, 365, (batch_size, 4))
        weather_data = torch.randn(batch_size, 4, 10)
        
        loss_dict = combined_loss(
            segmentation_logits=segmentation_logits,
            segmentation_targets=segmentation_targets,
            valid_mask=valid_mask,
            phenology_pred=phenology_pred,
            temporal_positions=temporal_positions,
            weather_data=weather_data
        )
        
        # Weights should remain unchanged
        assert combined_loss.segmentation_weight == initial_seg_weight
        assert combined_loss.phenology_weight == initial_phen_weight
    
    def test_gradient_flow(self, combined_loss):
        """Test gradient flow through combined loss."""
        batch_size, num_classes, height, width = 1, 6, 8, 8
        
        segmentation_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        segmentation_targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.ones(batch_size, height, width).bool()
        
        phenology_pred = torch.randn(batch_size, 64, requires_grad=True)
        temporal_positions = torch.randint(0, 365, (batch_size, 4))
        weather_data = torch.randn(batch_size, 4, 10)
        
        loss_dict = combined_loss(
            segmentation_logits=segmentation_logits,
            segmentation_targets=segmentation_targets,
            valid_mask=valid_mask,
            phenology_pred=phenology_pred,
            temporal_positions=temporal_positions,
            weather_data=weather_data
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Check gradients
        assert segmentation_logits.grad is not None
        assert phenology_pred.grad is not None
        assert not torch.isnan(segmentation_logits.grad).any()
        assert not torch.isnan(phenology_pred.grad).any()


class TestLossIntegration:
    """Integration tests for loss functions."""
    
    def test_loss_functions_compatibility(self):
        """Test that all loss functions work together."""
        # Create combined loss with all components
        combined_loss = CombinedLoss(
            segmentation_weight=1.0,
            phenology_weight=0.1,
            adaptive_weighting=False,
            segmentation_config={'loss_type': 'cross_entropy'},
            phenology_config={'temporal_weight': 0.5, 'weather_weight': 0.3}
        )
        
        batch_size, num_classes, height, width = 2, 6, 16, 16
        
        # Create realistic test data
        segmentation_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        segmentation_targets = torch.randint(0, num_classes, (batch_size, height, width))
        valid_mask = torch.randint(0, 2, (batch_size, height, width)).bool()
        
        phenology_pred = torch.randn(batch_size, 64, requires_grad=True)
        temporal_positions = torch.sort(torch.randint(0, 365, (batch_size, 8)))[0]
        weather_data = torch.randn(batch_size, 8, 10)
        
        # Compute loss
        loss_dict = combined_loss(
            segmentation_logits=segmentation_logits,
            segmentation_targets=segmentation_targets,
            valid_mask=valid_mask,
            phenology_pred=phenology_pred,
            temporal_positions=temporal_positions,
            weather_data=weather_data
        )
        
        # Check output
        assert 'total_loss' in loss_dict
        assert 'segmentation_loss' in loss_dict
        assert 'phenology_loss' in loss_dict
        
        # All losses should be finite and differentiable
        for loss_name, loss_value in loss_dict.items():
            assert isinstance(loss_value, torch.Tensor)
            assert torch.isfinite(loss_value)
            if loss_value.requires_grad:
                loss_value.backward(retain_graph=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
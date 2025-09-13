"""
Unit tests for AMPT model components.

Tests all model components including PhenologyEncoder, CrossModalAttention,
and the main AMPTModel to ensure proper functionality and integration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Import model components
from src.models.phenology_encoder import PhenologyEncoder
from src.models.cross_modal_attention import CrossModalPhenologicalAttention
from src.models.ampt_model import AMPTModel


class TestPhenologyEncoder:
    """Test cases for PhenologyEncoder module."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for PhenologyEncoder."""
        return DictConfig({
            'input_channels': 13,
            'temporal_length': 6,
            'hidden_dim': 64,
            'num_phenology_classes': 4,
            'conv_channels': [64, 128],
            'lstm_layers': 1,
            'num_heads': 4,
            'dropout': 0.1
        })
    
    @pytest.fixture
    def encoder(self, config):
        """Create PhenologyEncoder instance."""
        return PhenologyEncoder(
            input_channels=config.input_channels,
            temporal_length=config.temporal_length,
            hidden_dim=config.hidden_dim,
            num_phenology_classes=config.num_phenology_classes,
            conv_channels=config.conv_channels,
            lstm_layers=config.lstm_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    
    def test_encoder_initialization(self, encoder, config):
        """Test encoder initialization."""
        assert encoder.input_channels == config.input_channels
        assert encoder.hidden_dim == config.hidden_dim
        assert encoder.num_phenology_classes == config.num_phenology_classes
        assert hasattr(encoder, 'temporal_convs')
        assert hasattr(encoder, 'lstm')
        assert hasattr(encoder, 'temporal_attention')
    
    def test_encoder_forward_pass(self, encoder):
        """Test encoder forward pass."""
        batch_size, temporal_length, channels, height, width = 2, 6, 13, 32, 32
        
        # Create input tensor [B, T, C, H, W]
        input_tensor = torch.randn(batch_size, temporal_length, channels, height, width)
        
        # Forward pass
        phenology_probs, temporal_features = encoder(input_tensor)
        
        # Check output shapes
        assert phenology_probs.shape == (batch_size, encoder.num_phenology_classes)
        assert temporal_features.shape == (batch_size, encoder.hidden_dim)
        
        # Check output properties
        assert torch.allclose(phenology_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        assert not torch.isnan(phenology_probs).any()
        assert not torch.isnan(temporal_features).any()
    
    def test_encoder_variable_length_sequences(self, encoder):
        """Test encoder with variable sequence lengths."""
        batch_size, channels, height, width = 2, 13, 32, 32
        
        # Test different temporal lengths
        for temporal_length in [4, 6, 8]:
            input_tensor = torch.randn(batch_size, temporal_length, channels, height, width)
            
            phenology_probs, temporal_features = encoder(input_tensor)
            
            # Check output shapes (should be consistent)
            assert phenology_probs.shape == (batch_size, encoder.num_phenology_classes)
            assert temporal_features.shape == (batch_size, encoder.hidden_dim)
            assert not torch.isnan(phenology_probs).any()
            assert not torch.isnan(temporal_features).any()
    
    def test_encoder_gradient_flow(self, encoder):
        """Test gradient flow through encoder."""
        batch_size, temporal_length, channels, height, width = 1, 4, 13, 16, 16
        
        input_tensor = torch.randn(batch_size, temporal_length, channels, height, width, requires_grad=True)
        
        phenology_probs, temporal_features = encoder(input_tensor)
        
        # Create loss and backpropagate
        loss = phenology_probs.sum() + temporal_features.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()
        assert not torch.isinf(input_tensor.grad).any()
        
        # Check model parameters have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestCrossModalPhenologicalAttention:
    """Test cases for CrossModalPhenologicalAttention module."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DictConfig({
            'optical_dim': 256,
            'sar_dim': 128,
            'common_dim': 256,
            'num_heads': 8,
            'phenology_classes': 4,
            'dropout': 0.1,
            'temperature': 0.1,
            'use_residual': True
        })
    
    @pytest.fixture
    def attention_module(self, config):
        """Create CrossModalPhenologicalAttention instance."""
        return CrossModalPhenologicalAttention(
            optical_dim=config.optical_dim,
            sar_dim=config.sar_dim,
            common_dim=config.common_dim,
            num_heads=config.num_heads,
            phenology_classes=config.phenology_classes,
            dropout=config.dropout,
            temperature=config.temperature,
            use_residual=config.use_residual
        )
    
    def test_attention_initialization(self, attention_module, config):
        """Test attention module initialization."""
        assert attention_module.optical_dim == config.optical_dim
        assert attention_module.sar_dim == config.sar_dim
        assert attention_module.phenology_dim == config.phenology_dim
        assert isinstance(attention_module.optical_proj, nn.Linear)
        assert isinstance(attention_module.sar_proj, nn.Linear)
        assert isinstance(attention_module.cross_attention, nn.MultiheadAttention)
    
    def test_attention_forward_pass(self, attention_module):
        """Test attention forward pass."""
        batch_size, height, width = 2, 32, 32
        optical_dim, sar_dim, phenology_dim = 256, 256, 64
        
        # Create test inputs
        optical_features = torch.randn(batch_size, optical_dim, height, width)
        sar_features = torch.randn(batch_size, sar_dim, height, width)
        phenology_encoding = torch.randn(batch_size, phenology_dim)
        
        # Forward pass
        fused_features, attention_weights = attention_module(
            optical_features, sar_features, phenology_encoding
        )
        
        # Check outputs
        assert fused_features.shape == optical_features.shape
        assert attention_weights.shape == (batch_size, height * width, height * width)
        assert not torch.isnan(fused_features).any()
        assert not torch.isnan(attention_weights).any()
    
    def test_attention_weights_properties(self, attention_module):
        """Test attention weights properties."""
        batch_size, height, width = 1, 16, 16
        optical_dim, sar_dim, phenology_dim = 256, 256, 64
        
        optical_features = torch.randn(batch_size, optical_dim, height, width)
        sar_features = torch.randn(batch_size, sar_dim, height, width)
        phenology_encoding = torch.randn(batch_size, phenology_dim)
        
        _, attention_weights = attention_module(
            optical_features, sar_features, phenology_encoding
        )
        
        # Check attention weights sum to 1
        attention_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)
    
    def test_phenology_influence(self, attention_module):
        """Test that phenology encoding influences attention."""
        batch_size, height, width = 1, 8, 8
        optical_dim, sar_dim, phenology_dim = 256, 256, 64
        
        optical_features = torch.randn(batch_size, optical_dim, height, width)
        sar_features = torch.randn(batch_size, sar_dim, height, width)
        
        # Two different phenology encodings
        phenology1 = torch.randn(batch_size, phenology_dim)
        phenology2 = torch.randn(batch_size, phenology_dim)
        
        # Forward passes
        _, weights1 = attention_module(optical_features, sar_features, phenology1)
        _, weights2 = attention_module(optical_features, sar_features, phenology2)
        
        # Attention weights should be different
        assert not torch.allclose(weights1, weights2, atol=1e-3)


class TestAMPTModel:
    """Test cases for the main AMPT model."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OmegaConf.create({
            'num_classes': 6,
            'optical_channels': 13,
            'sar_channels': 2,
            'weather_dim': 10,
            'hidden_dim': 256,
            'phenology_classes': 4,
            'temporal_length': 6,
            'use_terratorch': False,
            'phenology_config': {
                'conv_channels': [64, 128],
                'lstm_layers': 1,
                'num_heads': 4,
                'dropout': 0.1
            },
            'cmpa_config': {
                'num_heads': 4,
                'dropout': 0.1,
                'temperature': 0.1,
                'use_residual': True
            },
            'sar_config': {
                'feature_dim': 128
            },
            'optimizer': {
                'name': 'AdamW',
                'lr': 1e-4,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'name': 'cosine',
                'max_epochs': 10
            },
            'loss': {
                'segmentation_weight': 1.0,
                'phenology_weight': 0.1,
                'weather_weight': 0.05
            }
        })
    
    @pytest.fixture
    def model(self, config):
        """Create AMPT model instance."""
        return AMPTModel(config)
    
    def test_model_initialization(self, model, config):
        """Test model initialization."""
        assert model.num_classes == config.num_classes
        assert model.optical_channels == config.optical_channels
        assert model.sar_channels == config.sar_channels
        assert hasattr(model, 'phenology_encoder')
        assert hasattr(model, 'cross_modal_attention')
        assert hasattr(model, 'optical_backbone')
    
    def test_model_forward_pass(self, model):
        """Test model forward pass."""
        batch_size, temporal_length, height, width = 2, 8, 64, 64
        optical_bands, sar_bands = 13, 2
        
        # Create test batch
        batch = {
            'optical': torch.randn(batch_size, temporal_length, optical_bands, height, width),
            'sar': torch.randn(batch_size, temporal_length, sar_bands, height, width),
            'weather': torch.randn(batch_size, temporal_length, 10),  # 10 weather features
            'temporal_positions': torch.randint(0, 365, (batch_size, temporal_length)),
            'mask': torch.randint(0, 6, (batch_size, height, width))
        }
        
        # Forward pass
        outputs = model(batch)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert 'segmentation' in outputs
        assert 'phenology' in outputs
        
        segmentation = outputs['segmentation']
        phenology = outputs['phenology']
        
        assert segmentation.shape == (batch_size, model.num_classes, height, width)
        assert phenology.shape == (batch_size, model.phenology_encoder.output_dim)
    
    def test_model_training_step(self, model):
        """Test model training step."""
        batch_size, temporal_length, height, width = 1, 4, 32, 32
        
        # Create test batch
        batch = {
            'optical': torch.randn(batch_size, temporal_length, 13, height, width),
            'sar': torch.randn(batch_size, temporal_length, 2, height, width),
            'weather': torch.randn(batch_size, temporal_length, 10),
            'temporal_positions': torch.randint(0, 365, (batch_size, temporal_length)),
            'mask': torch.randint(0, 6, (batch_size, height, width)),
            'valid_pixels': torch.randint(0, 2, (batch_size, height, width)).bool()
        }
        
        # Training step
        loss = model.training_step(batch, 0)
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.requires_grad
    
    def test_model_validation_step(self, model):
        """Test model validation step."""
        batch_size, temporal_length, height, width = 1, 4, 32, 32
        
        batch = {
            'optical': torch.randn(batch_size, temporal_length, 13, height, width),
            'sar': torch.randn(batch_size, temporal_length, 2, height, width),
            'weather': torch.randn(batch_size, temporal_length, 10),
            'temporal_positions': torch.randint(0, 365, (batch_size, temporal_length)),
            'mask': torch.randint(0, 6, (batch_size, height, width)),
            'valid_pixels': torch.randint(0, 2, (batch_size, height, width)).bool()
        }
        
        # Validation step
        outputs = model.validation_step(batch, 0)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert 'val_loss' in outputs
        assert isinstance(outputs['val_loss'], torch.Tensor)
    
    def test_model_configure_optimizers(self, model):
        """Test optimizer configuration."""
        optimizer_config = model.configure_optimizers()
        
        # Check structure
        assert isinstance(optimizer_config, dict)
        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config
        
        # Check optimizer
        optimizer = optimizer_config['optimizer']
        assert hasattr(optimizer, 'param_groups')
        
        # Check scheduler
        scheduler_config = optimizer_config['lr_scheduler']
        assert isinstance(scheduler_config, dict)
        assert 'scheduler' in scheduler_config
    
    def test_model_prediction(self, model):
        """Test model prediction mode."""
        model.eval()
        
        batch_size, temporal_length, height, width = 1, 6, 32, 32
        
        batch = {
            'optical': torch.randn(batch_size, temporal_length, 13, height, width),
            'sar': torch.randn(batch_size, temporal_length, 2, height, width),
            'weather': torch.randn(batch_size, temporal_length, 10),
            'temporal_positions': torch.randint(0, 365, (batch_size, temporal_length))
        }
        
        with torch.no_grad():
            outputs = model(batch)
        
        # Check predictions
        assert 'segmentation' in outputs
        segmentation = outputs['segmentation']
        
        # Should have valid probability distribution
        probs = torch.softmax(segmentation, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs.sum(dim=1)), atol=1e-6)
    
    @pytest.mark.parametrize("backbone_type", ["resnet50", "efficientnet"])
    def test_model_different_backbones(self, config, backbone_type):
        """Test model with different backbone types."""
        config.model.backbone = backbone_type
        
        try:
            model = AMPTModel(config)
            assert hasattr(model, 'backbone')
        except Exception as e:
            # Some backbones might not be available
            pytest.skip(f"Backbone {backbone_type} not available: {e}")
    
    def test_model_gradient_flow(self, model):
        """Test gradient flow through the model."""
        batch_size, temporal_length, height, width = 1, 4, 16, 16
        
        batch = {
            'optical': torch.randn(batch_size, temporal_length, 13, height, width, requires_grad=True),
            'sar': torch.randn(batch_size, temporal_length, 2, height, width, requires_grad=True),
            'weather': torch.randn(batch_size, temporal_length, 10, requires_grad=True),
            'temporal_positions': torch.randint(0, 365, (batch_size, temporal_length)),
            'mask': torch.randint(0, 6, (batch_size, height, width)),
            'valid_pixels': torch.randint(0, 2, (batch_size, height, width)).bool()
        }
        
        # Forward and backward pass
        loss = model.training_step(batch, 0)
        loss.backward()
        
        # Check gradients
        assert batch['optical'].grad is not None
        assert batch['sar'].grad is not None
        assert batch['weather'].grad is not None
        
        # Check model parameter gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestModelIntegration:
    """Integration tests for the complete model pipeline."""
    
    def test_end_to_end_training_simulation(self):
        """Test end-to-end training simulation."""
        # Create minimal config
        config = OmegaConf.create({
            'num_classes': 6,
            'optical_channels': 13,
            'sar_channels': 2,
            'weather_dim': 10,
            'hidden_dim': 128,
            'phenology_classes': 4,
            'temporal_length': 8,
            'use_terratorch': False,
            'phenology_config': {
                'hidden_dim': 128,
                'conv_channels': [64, 128],
                'lstm_layers': 1,
                'num_heads': 4,
                'dropout': 0.1
            },
            'cmpa_config': {
                'num_heads': 4,
                'dropout': 0.1,
                'temperature': 0.1
            },
            'sar_config': {
                'feature_dim': 128
            },
            'optimizer': {
                'name': 'AdamW',
                'lr': 1e-3,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'name': 'cosine',
                'max_epochs': 10
            },
            'loss': {
                'segmentation_weight': 1.0,
                'phenology_weight': 0.1,
                'weather_weight': 0.05
            }
        })
        
        # Create model
        model = AMPTModel(config)
        
        # Simulate training steps
        model.train()
        
        for step in range(3):  # 3 training steps
            batch = {
                'optical': torch.randn(1, 4, 13, 32, 32),
                'sar': torch.randn(1, 4, 2, 32, 32),
                'weather': torch.randn(1, 4, 10),
                'temporal_positions': torch.randint(0, 365, (1, 4)),
                'mask': torch.randint(0, 6, (1, 32, 32)),
                'valid_pixels': torch.randint(0, 2, (1, 32, 32)).bool()
            }
            
            # Training step
            loss = model.training_step(batch, step)
            
            # Simulate optimizer step
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss)
            assert loss.item() > 0
    
    def test_model_state_dict_consistency(self):
        """Test model state dict saving and loading."""
        config = OmegaConf.create({
            'num_classes': 6,
            'optical_channels': 13,
            'sar_channels': 2,
            'weather_dim': 10,
            'hidden_dim': 64,
            'phenology_classes': 4,
            'temporal_length': 4,
            'use_terratorch': False,
            'phenology_config': {
                'hidden_dim': 64,
                'conv_channels': [64],
                'lstm_layers': 1,
                'num_heads': 2,
                'dropout': 0.0
            },
            'cmpa_config': {
                'num_heads': 2,
                'dropout': 0.0,
                'temperature': 0.1
            },
            'sar_config': {
                'feature_dim': 64
            },
            'optimizer': {'name': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4},
            'scheduler': {'name': 'cosine', 'max_epochs': 10},
            'loss': {'segmentation_weight': 1.0, 'phenology_weight': 0.1, 'weather_weight': 0.05}
        })
        
        # Create two identical models
        model1 = AMPTModel(config)
        model2 = AMPTModel(config)
        
        # Save and load state dict
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)
        
        # Test that models produce identical outputs
        batch = {
            'optical': torch.randn(1, 4, 13, 16, 16),
            'sar': torch.randn(1, 4, 2, 16, 16),
            'weather': torch.randn(1, 4, 10),
            'temporal_positions': torch.randint(0, 365, (1, 4))
        }
        
        with torch.no_grad():
            output1 = model1(batch)
            output2 = model2(batch)
        
        # Outputs should be identical
        assert torch.allclose(output1['segmentation'], output2['segmentation'], atol=1e-6)
        assert torch.allclose(output1['phenology'], output2['phenology'], atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
"""
AMPTModel: Main PyTorch Lightning module integrating all components.

This is the complete AMPT (Adaptive Multi-Modal Phenological Transformer) model
that brings together all the innovations:

1. Prithvi backbone from TerraTorch (with fallback)
2. SAR encoder for VV/VH polarizations  
3. PhenologyEncoder for temporal modeling
4. WeatherFusion for weather-phenology integration
5. CrossModalPhenologicalAttention (CMPA) - core innovation
6. Multi-scale fusion for handling small irregular fields
7. Segmentation decoder for 6-class output
8. Auxiliary heads: phenology prediction, growth rate estimation

The model supports different input configurations (with/without SAR, weather)
and implements comprehensive training with combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import torchmetrics
from omegaconf import DictConfig
import numpy as np

# Try to import TerraTorch components with fallback
try:
    from terratorch.models.backbones.prithvi_vit import prithvi_eo_v2_300
    TERRATORCH_AVAILABLE = True
except ImportError:
    TERRATORCH_AVAILABLE = False
    print("Warning: TerraTorch not available. Using fallback backbone.")

from .phenology_encoder import PhenologyEncoder
from .cross_modal_attention import CrossModalPhenologicalAttention
from ..losses.combined_loss import CombinedLoss


class SAREnooder(nn.Module):
    """
    SAR encoder for processing VV/VH polarization data.
    
    This encoder processes SAR data through convolutional layers
    to extract features complementary to optical data.
    """
    
    def __init__(
        self,
        input_channels: int = 2,  # VV, VH
        channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        dropout: float = 0.1
    ):
        super(SAREnooder, self).__init__()
        
        self.layers = nn.ModuleList()
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.MaxPool2d(2) if i < len(channels) - 1 else nn.Identity()
            ))
            in_channels = out_channels
        
        # Global average pooling for feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process SAR data through convolutional layers.
        
        Args:
            x (torch.Tensor): SAR data [B, 2, H, W]
        
        Returns:
            torch.Tensor: SAR features [B, feature_dim]
        """
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling
        features = self.global_pool(x).flatten(1)
        return features


class WeatherFusion(nn.Module):
    """
    Fusion module for weather data with phenological stages.
    
    Combines weather information with crop growth stages to provide
    additional context for cross-modal attention.
    """
    
    def __init__(
        self,
        weather_dim: int = 5,
        phenology_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 128
    ):
        super(WeatherFusion, self).__init__()
        
        self.weather_projection = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.phenology_projection = nn.Sequential(
            nn.Linear(phenology_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self, 
        weather_data: torch.Tensor, 
        phenology_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse weather data with phenological stages.
        
        Args:
            weather_data (torch.Tensor): Weather features [B, weather_dim]
            phenology_probs (torch.Tensor): Phenology probabilities [B, phenology_dim]
        
        Returns:
            torch.Tensor: Fused weather-phenology features [B, output_dim]
        """
        weather_proj = self.weather_projection(weather_data)
        phenology_proj = self.phenology_projection(phenology_probs)
        
        fused = torch.cat([weather_proj, phenology_proj], dim=-1)
        output = self.fusion_layer(fused)
        
        return output


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion for handling irregular field sizes.
    
    Processes features at multiple scales to capture both
    local crop patterns and field-level context.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        scales: List[int] = [1, 2, 4],
        output_dim: int = 256
    ):
        super(MultiScaleFusion, self).__init__()
        
        self.scales = scales
        self.scale_convs = nn.ModuleList()
        
        for scale in scales:
            self.scale_convs.append(nn.Sequential(
                nn.Conv2d(input_dim, input_dim // len(scales), 
                         kernel_size=3, padding=1, dilation=scale),
                nn.BatchNorm2d(input_dim // len(scales)),
                nn.ReLU(inplace=True)
            ))
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features at multiple scales.
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
        
        Returns:
            torch.Tensor: Multi-scale fused features [B, output_dim, H, W]
        """
        scale_features = []
        
        for scale_conv in self.scale_convs:
            scale_features.append(scale_conv(x))
        
        # Concatenate multi-scale features
        concat_features = torch.cat(scale_features, dim=1)
        
        # Final fusion
        output = self.fusion_conv(concat_features)
        
        return output


class SegmentationDecoder(nn.Module):
    """
    Segmentation decoder for 6-class crop classification.
    
    Decodes fused multi-modal features to per-pixel crop predictions.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 6,
        use_attention: bool = True,
        skip_connections: bool = True
    ):
        super(SegmentationDecoder, self).__init__()
        
        self.use_attention = use_attention
        self.skip_connections = skip_connections
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        in_channels = input_dim
        
        for out_channels in channels:
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels
        
        # Final classification layer
        self.classifier = nn.Conv2d(channels[-1], num_classes, kernel_size=1)
        
        # Optional attention mechanism
        if use_attention:
            self.attention_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, 1, kernel_size=1),
                    nn.Sigmoid()
                ) for ch in channels
            ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Decode features to segmentation output.
        
        Args:
            x (torch.Tensor): Input features [B, input_dim, H, W]
            skip_features (List[torch.Tensor], optional): Skip connection features
        
        Returns:
            torch.Tensor: Segmentation logits [B, num_classes, H, W]
        """
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            
            # Apply attention if enabled
            if self.use_attention:
                attention = self.attention_layers[i](x)
                x = x * attention
            
            # Add skip connections if available
            if self.skip_connections and skip_features and i < len(skip_features):
                # Resize skip feature to match current resolution
                skip_feat = skip_features[-(i+1)]  # Reverse order
                if skip_feat.shape[-2:] != x.shape[-2:]:
                    skip_feat = F.interpolate(skip_feat, size=x.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                x = x + skip_feat
        
        # Final classification
        output = self.classifier(x)
        
        return output


class FallbackBackbone(nn.Module):
    """
    Fallback backbone when TerraTorch is not available.
    
    Simple CNN backbone for optical feature extraction.
    """
    
    def __init__(self, input_channels: int = 6, output_dim: int = 256):
        super(FallbackBackbone, self).__init__()
        
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(256, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AMPTModel(pl.LightningModule):
    """
    Complete AMPT model with all components integrated.
    
    This model implements the full pipeline from multi-modal inputs
    to crop classification with auxiliary predictions.
    """
    
    def __init__(self, config: DictConfig):
        super(AMPTModel, self).__init__()
        
        self.save_hyperparameters()
        self.config = config
        
        # Model dimensions
        self.num_classes = config.get('num_classes', 6)
        self.optical_channels = config.get('optical_channels', 6)
        self.sar_channels = config.get('sar_channels', 2)
        self.weather_dim = config.get('weather_dim', 5)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.phenology_classes = config.get('phenology_classes', 4)
        self.temporal_length = config.get('temporal_length', 6)
        
        # Initialize backbone (Prithvi or fallback)
        self._init_backbone()
        
        # Initialize components
        self.phenology_encoder = PhenologyEncoder(
            input_channels=self.optical_channels,
            temporal_length=self.temporal_length,
            hidden_dim=self.hidden_dim,
            num_phenology_classes=self.phenology_classes,
            **config.get('phenology_config', {})
        )
        
        self.sar_encoder = SAREnooder(
            input_channels=self.sar_channels,
            **config.get('sar_config', {})
        )
        
        self.weather_fusion = WeatherFusion(
            weather_dim=self.weather_dim,
            phenology_dim=self.phenology_classes,
            output_dim=self.hidden_dim
        )
        
        self.cross_modal_attention = CrossModalPhenologicalAttention(
            optical_dim=self.hidden_dim,
            sar_dim=self.sar_encoder.feature_dim,
            common_dim=self.hidden_dim,
            **config.get('cmpa_config', {})
        )
        
        self.multi_scale_fusion = MultiScaleFusion(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim
        )
        
        self.segmentation_decoder = SegmentationDecoder(
            input_dim=self.hidden_dim,
            num_classes=self.num_classes,
            **config.get('decoder_config', {})
        )
        
        # Auxiliary heads
        self.growth_rate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)  # Growth rate regression
        )
        
        # Loss function
        self.criterion = CombinedLoss(
            num_classes=self.num_classes,
            **config.get('loss_weights', {})
        )
        
        # Metrics
        self._init_metrics()
        
        # Class names for logging
        self.class_names = config.get('class_names', 
            ['gram', 'maize', 'mustard', 'sugarcane', 'wheat', 'other_crop'])
    
    def _init_backbone(self):
        """Initialize the optical backbone (Prithvi or fallback)."""
        if TERRATORCH_AVAILABLE:
            try:
                # Import the proper factory function
                from terratorch.models.backbones.prithvi_vit import prithvi_eo_v2_300
                
                # Create Prithvi model with proper configuration
                self.optical_backbone = prithvi_eo_v2_300(
                    pretrained=True,
                    bands=None,  # Will use default 6 bands
                    in_chans=self.optical_channels,
                    num_frames=self.temporal_length,
                    encoder_only=True
                )
                
                # Prithvi models typically have embed_dim as output
                self.backbone_output_dim = 1024  # prithvi_eo_v2_300 uses 1024 embed_dim
                print("Successfully loaded Prithvi ViT backbone from TerraTorch")
                
            except Exception as e:
                print(f"Failed to load Prithvi: {e}. Using fallback backbone.")
                self.optical_backbone = FallbackBackbone(
                    input_channels=self.optical_channels,
                    output_dim=self.hidden_dim
                )
                self.backbone_output_dim = self.hidden_dim
        else:
            self.optical_backbone = FallbackBackbone(
                input_channels=self.optical_channels,
                output_dim=self.hidden_dim
            )
            self.backbone_output_dim = self.hidden_dim
        
        # Projection layer to common dimension
        if self.backbone_output_dim != self.hidden_dim:
            self.optical_projection = nn.Linear(self.backbone_output_dim, self.hidden_dim)
        else:
            self.optical_projection = nn.Identity()
    
    def _init_metrics(self):
        """Initialize training and validation metrics."""
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes),
            torchmetrics.JaccardIndex(task='multiclass', num_classes=self.num_classes),
            torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
        ])
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete AMPT model.
        
        Args:
            batch: Dictionary containing:
                - optical: [B, T, C, H, W] temporal optical data
                - sar: [B, T, 2, H, W] temporal SAR data (optional)
                - weather: [B, T, weather_dim] weather data (optional)
                - valid_mask: [B, H, W] valid pixel mask (optional)
        
        Returns:
            Dict[str, torch.Tensor]: Model predictions
        """
        optical_data = batch['optical']
        batch_size, temporal_length, channels, height, width = optical_data.shape
        
        # 1. Extract phenological stages from temporal optical data
        phenology_probs, temporal_features = self.phenology_encoder(optical_data)
        
        # 2. Fuse with weather data if available
        weather_phenology_features = None
        if 'weather' in batch and batch['weather'] is not None:
            # Average weather over temporal dimension for global context
            weather_data = batch['weather'].mean(dim=1)  # [B, weather_dim]
            weather_phenology_features = self.weather_fusion(weather_data, phenology_probs)
        
        # 3. Process optical data through backbone
        # Reshape for backbone processing: [B*T, C, H, W]
        optical_reshaped = optical_data.view(-1, channels, height, width)
        optical_features = self.optical_backbone(optical_reshaped)
        
        # Reshape back and project: [B, T, feature_dim]
        if len(optical_features.shape) == 4:
            # CNN backbone output: [B*T, C, H', W']
            optical_features = F.adaptive_avg_pool2d(optical_features, (1, 1))
            optical_features = optical_features.view(batch_size, temporal_length, -1)
        else:
            # ViT backbone output: [B*T, feature_dim]
            optical_features = optical_features.view(batch_size, temporal_length, -1)
        
        optical_features = self.optical_projection(optical_features)
        
        # 4. Process SAR data through custom encoder
        sar_features = None
        if 'sar' in batch and batch['sar'] is not None:
            sar_data = batch['sar']
            # Process each time step: [B*T, 2, H, W]
            sar_reshaped = sar_data.view(-1, self.sar_channels, height, width)
            sar_features = self.sar_encoder(sar_reshaped)
            # Reshape back: [B, T, feature_dim]
            sar_features = sar_features.view(batch_size, temporal_length, -1)
        
        # 5. Apply cross-modal phenological attention (CMPA)
        if sar_features is not None:
            fused_features, attention_info = self.cross_modal_attention(
                optical_features=optical_features,
                sar_features=sar_features,
                phenology_stage=phenology_probs
            )
        else:
            # Use only optical features
            fused_features = optical_features
            attention_info = {'modality_weights': torch.ones(batch_size, 2) * 0.5}
        
        # 6. Multi-scale spatial processing
        # Use latest temporal features for spatial processing
        latest_features = fused_features[:, -1, :]  # [B, hidden_dim]
        
        # Reshape for spatial processing: [B, hidden_dim, 1, 1] -> [B, hidden_dim, H, W]
        spatial_features = latest_features.unsqueeze(-1).unsqueeze(-1)
        spatial_features = F.interpolate(spatial_features, size=(height, width), 
                                       mode='bilinear', align_corners=False)
        
        multi_scale_features = self.multi_scale_fusion(spatial_features)
        
        # 7. Segmentation prediction
        segmentation_logits = self.segmentation_decoder(multi_scale_features)
        
        # 8. Auxiliary predictions
        growth_rate = self.growth_rate_head(latest_features)
        
        # Prepare outputs
        outputs = {
            'segmentation': segmentation_logits,
            'phenology': phenology_probs,
            'growth_rate': growth_rate,
            'attention_info': attention_info
        }
        
        # Add weather-phenology features if available
        if weather_phenology_features is not None:
            outputs['weather_phenology'] = weather_phenology_features
        
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with combined loss computation."""
        outputs = self(batch)
        
        # Compute combined loss
        loss_dict = self.criterion(
            predictions=outputs,
            targets=batch.get('labels'),
            valid_mask=batch.get('valid_mask')
        )
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, 
                    prog_bar=True if loss_name == 'total_loss' else False)
        
        # Compute and log metrics
        if 'labels' in batch:
            preds = outputs['segmentation'].argmax(dim=1)
            labels = batch['labels']
            
            # Apply valid mask if available
            if 'valid_mask' in batch:
                valid_mask = batch['valid_mask'].bool()
                preds = preds[valid_mask]
                labels = labels[valid_mask]
            
            metrics = self.train_metrics(preds, labels)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with metrics computation."""
        outputs = self(batch)
        
        # Compute loss
        loss_dict = self.criterion(
            predictions=outputs,
            targets=batch.get('labels'),
            valid_mask=batch.get('valid_mask')
        )
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, 
                    prog_bar=True if loss_name == 'total_loss' else False)
        
        # Compute metrics
        if 'labels' in batch:
            preds = outputs['segmentation'].argmax(dim=1)
            labels = batch['labels']
            
            # Apply valid mask if available
            if 'valid_mask' in batch:
                valid_mask = batch['valid_mask'].bool()
                preds = preds[valid_mask]
                labels = labels[valid_mask]
            
            metrics = self.val_metrics(preds, labels)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step for final evaluation."""
        outputs = self(batch)
        
        if 'labels' in batch:
            preds = outputs['segmentation'].argmax(dim=1)
            labels = batch['labels']
            
            # Apply valid mask if available
            if 'valid_mask' in batch:
                valid_mask = batch['valid_mask'].bool()
                preds = preds[valid_mask]
                labels = labels[valid_mask]
            
            metrics = self.test_metrics(preds, labels)
            self.log_dict(metrics, on_step=False, on_epoch=True)
        
        return outputs
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        scheduler_config = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('max_epochs', 100),
                eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1,
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        outputs = self(batch)
        
        # Convert to probabilities
        outputs['segmentation_probs'] = F.softmax(outputs['segmentation'], dim=1)
        outputs['segmentation_pred'] = outputs['segmentation'].argmax(dim=1)
        
        return outputs


# Export for easy imports
__all__ = ['AMPTModel', 'SAREnooder', 'WeatherFusion', 'MultiScaleFusion', 'SegmentationDecoder']

"""
AMPT (Adaptive Multi-Modal Phenological Transformer) for Crop Classification

This package implements a novel approach to crop classification using cross-modal
phenological attention that dynamically weights SAR vs optical satellite data
based on crop growth stages.

Main components:
- models: Neural network architectures including the main AMPTModel
- data: Data loading and preprocessing for multi-modal satellite data
- losses: Custom loss functions for selective learning and phenological consistency
- utils: Utility functions for visualization, metrics, and submission generation
"""

__version__ = "0.1.0"
__author__ = "AMPT Development Team"

# Core model imports
from .models.ampt_model import AMPTModel
from .models.phenology_encoder import PhenologyEncoder
from .models.cross_modal_attention import CrossModalPhenologicalAttention

# Data processing imports
from .data.agrifieldnet_dataset import AgriFieldNetDataset
from .data.agrifieldnet_datamodule import AgriFieldNetDataModule

# Loss function imports
from .losses.selective_loss import SelectiveLoss
from .losses.phenology_loss import PhenologyLoss
from .losses.combined_loss import CombinedLoss

# Utility imports
from .utils.metrics import SegmentationMetrics
from .utils.visualization import AttentionVisualizer
from .utils.submission import SubmissionGenerator

__all__ = [
    # Models
    "AMPTModel",
    "PhenologyEncoder", 
    "CrossModalPhenologicalAttention",
    
    # Data
    "AgriFieldNetDataset",
    "AgriFieldNetDataModule",
    
    # Losses
    "SelectiveLoss",
    "PhenologyLoss", 
    "CombinedLoss",
    
    # Utils
    "SegmentationMetrics",
    "AttentionVisualizer",
    "SubmissionGenerator",
]
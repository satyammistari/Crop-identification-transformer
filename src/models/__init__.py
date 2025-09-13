"""
Neural network models for AMPT crop classification.

This module contains:
- AMPTModel: Main Lightning module with all components
- PhenologyEncoder: Temporal encoding for crop growth stages  
- CrossModalPhenologicalAttention: Core innovation for dynamic cross-modal fusion
- SAR encoder and other auxiliary components
"""

# Core imports will be added as modules are created
__all__ = [
    "AMPTModel",
    "PhenologyEncoder", 
    "CrossModalPhenologicalAttention",
]
"""
Loss functions for AMPT crop classification.

This module provides:
- SelectiveLoss: Loss computation only on valid pixels
- PhenologyLoss: Phenological consistency and weather alignment
- CombinedLoss: Multi-objective loss combining all components
"""

__all__ = [
    "SelectiveLoss",
    "PhenologyLoss", 
    "CombinedLoss",
]

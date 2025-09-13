#!/usr/bin/env python3
"""
Submission creation script for AMPT model.

This script generates competition submissions by:
- Loading trained model from checkpoint
- Running inference on test data
- Converting predictions to submission format
- Creating submission files (CSV and raster)

Example usage:
    python scripts/create_submission.py checkpoint_path=outputs/checkpoints/best.ckpt
    python scripts/create_submission.py checkpoint_path=best.ckpt output_dir=submissions/
    python scripts/create_submission.py --config-name=submission_config
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import zipfile

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.ampt_model import AMPTModel
from data.datamodule import AgriFieldNetDataModule
from utils.submission import SubmissionGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubmissionCreator:
    """
    Competition submission creator.
    
    Handles the complete pipeline from model inference to submission file creation.
    """
    
    def __init__(self, model: AMPTModel, config: DictConfig):
        """
        Initialize submission creator.
        
        Args:
            model: Trained AMPT model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize submission generator
        self.submission_generator = SubmissionGenerator()
        
        logger.info("SubmissionCreator initialized")
    
    def generate_predictions(self, dataloader: torch.utils.data.DataLoader) -> Tuple[List[torch.Tensor], List[str], List[Dict]]:
        """
        Generate predictions for all test samples.
        
        Args:
            dataloader: Test data loader
            
        Returns:
            Tuple of (predictions, sample_ids, metadata)
        """
        logger.info("Generating predictions...")
        
        self.model.eval()
        all_predictions = []
        all_sample_ids = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Extract predictions (convert logits to class predictions)
                predictions = torch.argmax(outputs['segmentation_logits'], dim=1).cpu()
                all_predictions.append(predictions)
                
                # Extract sample IDs
                if 'image_id' in batch:
                    sample_ids = batch['image_id']
                else:
                    sample_ids = [f"test_{batch_idx}_{i}" for i in range(len(predictions))]
                all_sample_ids.extend(sample_ids)
                
                # Extract metadata if available
                metadata = []
                for i in range(len(predictions)):
                    meta = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'prediction_shape': predictions[i].shape,
                    }
                    
                    # Add geospatial info if available
                    if 'bounds' in batch:
                        meta['bounds'] = batch['bounds'][i] if isinstance(batch['bounds'], (list, tuple)) else batch['bounds']
                    if 'crs' in batch:
                        meta['crs'] = batch['crs'][i] if isinstance(batch['crs'], (list, tuple)) else batch['crs']
                    
                    metadata.append(meta)
                
                all_metadata.extend(metadata)
        
        # Concatenate predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        
        logger.info(f"Generated {len(all_predictions)} predictions")
        return all_predictions, all_sample_ids, all_metadata
    
    def create_csv_submission(self, predictions: torch.Tensor, sample_ids: List[str], 
                             output_path: Path) -> None:
        """
        Create CSV submission file.
        
        Args:
            predictions: Model predictions [N, H, W]
            sample_ids: List of sample identifiers
            output_path: Path to save CSV file
        """
        logger.info("Creating CSV submission...")
        
        submission_data = []
        
        for pred, sample_id in zip(predictions, sample_ids):
            # Convert prediction to numpy
            pred_np = pred.numpy().astype(np.uint8)
            
            # For each pixel, create a row (if required by competition format)
            # This example assumes a flattened format - adjust based on competition requirements
            height, width = pred_np.shape
            
            for y in range(height):
                for x in range(width):
                    submission_data.append({
                        'image_id': sample_id,
                        'x': x,
                        'y': y,
                        'class': pred_np[y, x]
                    })
        
        # Create DataFrame and save
        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"CSV submission saved to: {output_path}")
    
    def validate_predictions(self, predictions: torch.Tensor, 
                           expected_classes: int = 6) -> bool:
        """
        Validate prediction format and values.
        
        Args:
            predictions: Model predictions
            expected_classes: Expected number of classes
            
        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating predictions...")
        
        # Check data type
        if not isinstance(predictions, torch.Tensor):
            logger.error("Predictions must be torch.Tensor")
            return False
        
        # Check dimensions
        if len(predictions.shape) != 3:
            logger.error(f"Predictions must be 3D [N, H, W], got shape: {predictions.shape}")
            return False
        
        # Check value range
        min_val = predictions.min().item()
        max_val = predictions.max().item()
        
        if min_val < 0 or max_val >= expected_classes:
            logger.error(f"Prediction values must be in range [0, {expected_classes-1}], "
                        f"got range [{min_val}, {max_val}]")
            return False
        
        # Check for reasonable class distribution
        unique_classes = torch.unique(predictions)
        logger.info(f"Found {len(unique_classes)} unique classes: {unique_classes.tolist()}")
        
        # Log class distribution
        class_counts = torch.bincount(predictions.flatten(), minlength=expected_classes)
        total_pixels = predictions.numel()
        
        logger.info("Class distribution:")
        for i, count in enumerate(class_counts):
            percentage = 100.0 * count.item() / total_pixels
            logger.info(f"  Class {i}: {count} pixels ({percentage:.2f}%)")
        
        logger.info("Prediction validation passed")
        return True


def load_model_from_checkpoint(checkpoint_path: str, config: DictConfig) -> AMPTModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        
    Returns:
        Loaded AMPT model
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Load model
        model = AMPTModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config.model,
            strict=False
        )
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main submission creation function.
    
    Args:
        config: Hydra configuration
    """
    try:
        logger.info("Starting AMPT submission creation...")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
        
        # Validate checkpoint path
        checkpoint_path = config.get('checkpoint_path')
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be specified")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Setup output directory
        output_dir = Path(config.get('output_dir', 'outputs/submissions'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model = load_model_from_checkpoint(str(checkpoint_path), config)
        
        # Initialize data module for test data
        logger.info("Initializing test data...")
        datamodule = AgriFieldNetDataModule(config.data)
        datamodule.setup('test')
        test_dataloader = datamodule.test_dataloader()
        
        # Initialize submission creator
        submission_creator = SubmissionCreator(model, config)
        
        # Generate predictions
        predictions, sample_ids, metadata = submission_creator.generate_predictions(test_dataloader)
        
        # Validate predictions
        if not submission_creator.validate_predictions(predictions, config.model.get('num_classes', 6)):
            raise ValueError("Prediction validation failed")
        
        # Create CSV submission
        csv_path = output_dir / 'submission.csv'
        submission_creator.create_csv_submission(predictions, sample_ids, csv_path)
        
        # Log summary
        logger.info("Submission creation completed successfully!")
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info(f"Submission saved to: {csv_path}")
        
        # Log class distribution summary
        class_counts = torch.bincount(predictions.flatten(), minlength=config.model.get('num_classes', 6))
        total_pixels = predictions.numel()
        logger.info("Final class distribution:")
        for i, count in enumerate(class_counts):
            percentage = 100.0 * count.item() / total_pixels
            logger.info(f"  Class {i}: {percentage:.2f}%")
        
    except Exception as e:
        logger.error(f"Submission creation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
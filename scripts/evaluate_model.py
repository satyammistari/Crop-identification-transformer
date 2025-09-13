#!/usr/bin/env python3
"""
Evaluation script for AMPT model.

This script handles model evaluation including:
- Loading trained model from checkpoint
- Running inference on test data
- Computing comprehensive metrics
- Generating evaluation reports
- Visualizing predictions and attention maps

Example usage:
    python scripts/evaluate_model.py checkpoint_path=outputs/checkpoints/best.ckpt
    python scripts/evaluate_model.py checkpoint_path=best.ckpt data.test_dir=data/test
    python scripts/evaluate_model.py --config-name=eval_config
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.ampt_model import AMPTModel
from data.datamodule import AgriFieldNetDataModule
from utils.metrics import SegmentationMetrics
from utils.visualizer import AttentionVisualizer
from utils.submission import SubmissionGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Handles all aspects of model evaluation including metrics computation,
    visualization generation, and report creation.
    """
    
    def __init__(self, model: AMPTModel, config: DictConfig):
        """
        Initialize evaluator.
        
        Args:
            model: Trained AMPT model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize components
        self.metrics = SegmentationMetrics(
            num_classes=config.model.get('num_classes', 6),
            ignore_index=config.get('ignore_index', -1)
        )
        
        self.visualizer = AttentionVisualizer()
        self.submission_generator = SubmissionGenerator()
        
        # Results storage
        self.results = {
            'predictions': [],
            'targets': [],
            'metrics': {},
            'attention_maps': [],
            'sample_ids': []
        }
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_dataset(self, dataloader: torch.utils.data.DataLoader, 
                        split_name: str = 'test') -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            split_name: Name of the dataset split
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating on {split_name} set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_valid_masks = []
        all_attention_maps = []
        all_sample_ids = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Extract predictions and targets
                predictions = outputs['segmentation_logits'].cpu()
                targets = batch['mask'].cpu()
                valid_mask = batch.get('valid_pixels', torch.ones_like(targets).bool()).cpu()
                
                # Store results
                all_predictions.append(predictions)
                all_targets.append(targets)
                all_valid_masks.append(valid_mask)
                
                # Extract attention maps if available
                if 'attention_weights' in outputs:
                    attention_maps = outputs['attention_weights'].cpu()
                    all_attention_maps.append(attention_maps)
                
                # Store sample IDs
                if 'image_id' in batch:
                    all_sample_ids.extend(batch['image_id'])
                else:
                    all_sample_ids.extend([f"{split_name}_{batch_idx}_{i}" 
                                          for i in range(len(predictions))])
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_valid_masks = torch.cat(all_valid_masks, dim=0)
        
        if all_attention_maps:
            all_attention_maps = torch.cat(all_attention_maps, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets, all_valid_masks)
        
        # Store results
        split_results = {
            'predictions': all_predictions,
            'targets': all_targets,
            'valid_masks': all_valid_masks,
            'attention_maps': all_attention_maps if all_attention_maps else None,
            'sample_ids': all_sample_ids,
            'metrics': metrics
        }
        
        self.results[split_name] = split_results
        
        logger.info(f"Evaluation completed for {split_name} set")
        logger.info(f"Overall mIoU: {metrics['miou']:.4f}")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        
        return split_results
    
    def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                        valid_masks: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions [N, C, H, W]
            targets: Ground truth targets [N, H, W]
            valid_masks: Valid pixel masks [N, H, W]
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert predictions to class indices
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Apply valid mask
        valid_pred = pred_classes[valid_masks]
        valid_targets = targets[valid_masks]
        
        # Compute per-class IoU
        num_classes = predictions.shape[1]
        per_class_iou = []
        per_class_accuracy = []
        
        for class_idx in range(num_classes):
            pred_mask = (valid_pred == class_idx)
            target_mask = (valid_targets == class_idx)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                per_class_iou.append(iou.item())
            else:
                per_class_iou.append(float('nan'))
            
            # Per-class accuracy
            if target_mask.sum() > 0:
                accuracy = (pred_mask & target_mask).sum().float() / target_mask.sum().float()
                per_class_accuracy.append(accuracy.item())
            else:
                per_class_accuracy.append(float('nan'))
        
        # Overall metrics
        overall_accuracy = (valid_pred == valid_targets).float().mean().item()
        miou = np.nanmean(per_class_iou)
        
        # Pixel-weighted accuracy
        pixel_weights = torch.bincount(valid_targets, minlength=num_classes).float()
        pixel_weights = pixel_weights / pixel_weights.sum()
        weighted_accuracy = sum(acc * weight for acc, weight in 
                               zip(per_class_accuracy, pixel_weights) if not np.isnan(acc))
        
        return {
            'miou': miou,
            'accuracy': overall_accuracy,
            'weighted_accuracy': weighted_accuracy,
            'per_class_iou': per_class_iou,
            'per_class_accuracy': per_class_accuracy,
            'num_samples': len(predictions),
            'num_valid_pixels': valid_masks.sum().item()
        }
    
    def create_evaluation_report(self, output_dir: Path) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Dictionary containing the full report
        """
        logger.info("Creating evaluation report...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'model_config': OmegaConf.to_container(self.config.model, resolve=True),
            'evaluation_config': OmegaConf.to_container(self.config, resolve=True),
            'results': {}
        }
        
        # Process each split
        for split_name, split_results in self.results.items():
            if isinstance(split_results, dict) and 'metrics' in split_results:
                split_report = self._create_split_report(split_name, split_results, output_dir)
                report['results'][split_name] = split_report
        
        # Save report as JSON
        report_path = output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        return report
    
    @staticmethod
    def _json_serializer(obj):
        """JSON serializer for numpy and torch objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
    Main evaluation function.
    
    Args:
        config: Hydra configuration
    """
    try:
        logger.info("Starting AMPT model evaluation...")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
        
        # Validate checkpoint path
        checkpoint_path = config.get('checkpoint_path')
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be specified")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Setup output directory
        output_dir = Path(config.get('output_dir', 'outputs/evaluation'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model = load_model_from_checkpoint(str(checkpoint_path), config)
        
        # Initialize data module
        logger.info("Initializing data module...")
        datamodule = AgriFieldNetDataModule(config.data)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model, config)
        
        # Evaluate on different splits
        splits_to_evaluate = config.get('splits_to_evaluate', ['test'])
        
        for split in splits_to_evaluate:
            logger.info(f"Setting up {split} data...")
            datamodule.setup(split)
            
            if split == 'test':
                dataloader = datamodule.test_dataloader()
            elif split == 'val':
                dataloader = datamodule.val_dataloader()
            elif split == 'train':
                dataloader = datamodule.train_dataloader()
            else:
                logger.warning(f"Unknown split: {split}, skipping...")
                continue
            
            # Run evaluation
            results = evaluator.evaluate_dataset(dataloader, split)
            
            logger.info(f"Results for {split} set:")
            logger.info(f"  mIoU: {results['metrics']['miou']:.4f}")
            logger.info(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
            logger.info(f"  Samples: {results['metrics']['num_samples']}")
        
        # Generate comprehensive report
        report = evaluator.create_evaluation_report(output_dir)
        
        # Generate submission file if requested
        if config.get('generate_submission', False) and 'test' in evaluator.results:
            logger.info("Generating submission file...")
            test_results = evaluator.results['test']
            submission_path = output_dir / 'submission.csv'
            
            evaluator.submission_generator.create_submission(
                predictions=test_results['predictions'],
                sample_ids=test_results['sample_ids'],
                output_path=submission_path
            )
            
            logger.info(f"Submission file saved to: {submission_path}")
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
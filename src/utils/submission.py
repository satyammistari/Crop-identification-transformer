"""
SubmissionGenerator: Competition submission utilities.

This module provides utilities for generating competition submissions including:
1. Model inference on test data
2. Prediction post-processing
3. Submission file generation (.tif and .csv formats)
4. Batch processing optimization
5. Result validation and quality checks
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """
    Generate competition submissions from trained AMPT model.
    
    This class handles the complete pipeline from model inference
    to submission file generation, including post-processing and
    format conversion.
    
    Args:
        model: Trained AMPT model
        device (str): Device for inference ('cuda' or 'cpu')
        batch_size (int): Batch size for inference
        num_classes (int): Number of crop classes
        class_names (List[str]): Names of crop classes
        use_tta (bool): Whether to use test-time augmentation
        tta_scales (List[float]): Scales for TTA
        tta_flips (List[str]): Flip augmentations for TTA
    """
    
    def __init__(
        self,
        model,
        device: str = 'cuda',
        batch_size: int = 4,
        num_classes: int = 6,
        class_names: Optional[List[str]] = None,
        use_tta: bool = True,
        tta_scales: Optional[List[float]] = None,
        tta_flips: Optional[List[str]] = None
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.use_tta = use_tta
        
        # Default class names
        if class_names is None:
            self.class_names = ['gram', 'maize', 'mustard', 'sugarcane', 'wheat', 'other_crop']
        else:
            self.class_names = class_names
        
        # TTA parameters
        if tta_scales is None:
            self.tta_scales = [1.0, 0.75, 1.25]
        else:
            self.tta_scales = tta_scales
        
        if tta_flips is None:
            self.tta_flips = ['none', 'horizontal', 'vertical', 'both']
        else:
            self.tta_flips = tta_flips
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        logger.info(f"SubmissionGenerator initialized with device={device}, "
                   f"batch_size={batch_size}, TTA={'enabled' if use_tta else 'disabled'}")
    
    def generate_submission(
        self,
        dataloader,
        output_dir: str,
        submission_name: str = 'submission',
        save_predictions: bool = True,
        save_probabilities: bool = False,
        apply_crf: bool = False,
        ensemble_weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete submission from dataloader.
        
        Args:
            dataloader: PyTorch DataLoader for test data
            output_dir (str): Directory to save submission files
            submission_name (str): Name prefix for submission files
            save_predictions (bool): Whether to save prediction .tif files
            save_probabilities (bool): Whether to save probability maps
            apply_crf (bool): Whether to apply CRF post-processing
            ensemble_weights (List[float], optional): Weights for ensemble prediction
        
        Returns:
            Dict[str, Any]: Submission results and statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting submission generation to {output_dir}")
        
        # Storage for results
        predictions = {}
        probabilities = {}
        image_ids = []
        confidence_scores = []
        
        # Process batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Get predictions
                if self.use_tta:
                    pred_probs, pred_labels = self._predict_with_tta(batch)
                else:
                    pred_probs, pred_labels = self._predict_single(batch)
                
                # Apply post-processing
                if apply_crf:
                    pred_labels = self._apply_crf_postprocessing(
                        batch.get('optical', batch.get('image')), 
                        pred_probs, 
                        pred_labels
                    )
                
                # Store results
                for i in range(len(pred_labels)):
                    image_id = batch['image_id'][i]
                    image_ids.append(image_id)
                    
                    predictions[image_id] = pred_labels[i].cpu().numpy()
                    
                    if save_probabilities:
                        probabilities[image_id] = pred_probs[i].cpu().numpy()
                    
                    # Calculate confidence score
                    confidence = self._calculate_confidence(pred_probs[i])
                    confidence_scores.append(confidence)
        
        logger.info(f"Generated predictions for {len(predictions)} images")
        
        # Save prediction files
        if save_predictions:
            pred_dir = output_dir / 'predictions'
            pred_dir.mkdir(exist_ok=True)
            
            self._save_prediction_files(predictions, pred_dir)
        
        if save_probabilities:
            prob_dir = output_dir / 'probabilities'
            prob_dir.mkdir(exist_ok=True)
            
            self._save_probability_files(probabilities, prob_dir)
        
        # Generate submission CSV
        submission_csv = output_dir / f'{submission_name}.csv'
        self._generate_submission_csv(predictions, submission_csv)
        
        # Generate submission statistics
        stats = self._generate_submission_stats(
            predictions, confidence_scores, image_ids
        )
        
        # Save statistics
        stats_file = output_dir / f'{submission_name}_stats.json'
        self._save_submission_stats(stats, stats_file)
        
        logger.info(f"Submission generation completed. Files saved to {output_dir}")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities if save_probabilities else None,
            'stats': stats,
            'submission_file': str(submission_csv),
            'output_dir': str(output_dir)
        }
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _predict_single(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single prediction without TTA.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probabilities and predictions
        """
        outputs = self.model(batch)
        
        if isinstance(outputs, dict):
            logits = outputs['segmentation']
        else:
            logits = outputs
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return probs, preds
    
    def _predict_with_tta(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with Test-Time Augmentation.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Averaged probabilities and predictions
        """
        batch_size = len(batch['image_id'])
        height, width = batch['optical'].shape[-2:]
        
        # Accumulate predictions
        total_probs = torch.zeros(batch_size, self.num_classes, height, width, device=self.device)
        total_weight = 0
        
        # Apply different augmentations
        for scale in self.tta_scales:
            for flip_type in self.tta_flips:
                # Apply augmentation
                aug_batch = self._apply_tta_augmentation(batch, scale, flip_type)
                
                # Get prediction
                probs, _ = self._predict_single(aug_batch)
                
                # Reverse augmentation
                probs = self._reverse_tta_augmentation(probs, scale, flip_type, height, width)
                
                # Accumulate
                total_probs += probs
                total_weight += 1
        
        # Average predictions
        avg_probs = total_probs / total_weight
        avg_preds = torch.argmax(avg_probs, dim=1)
        
        return avg_probs, avg_preds
    
    def _apply_tta_augmentation(
        self, 
        batch: Dict[str, torch.Tensor], 
        scale: float, 
        flip_type: str
    ) -> Dict[str, torch.Tensor]:
        """Apply TTA augmentation to batch."""
        aug_batch = {}
        
        for key, value in batch.items():
            if key == 'optical' and isinstance(value, torch.Tensor):
                # Apply scale and flip to optical data
                aug_value = value.clone()
                
                # Scale
                if scale != 1.0:
                    _, _, t, c, h, w = aug_value.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    aug_value = F.interpolate(
                        aug_value.view(-1, c, h, w), 
                        size=(new_h, new_w), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    aug_value = aug_value.view(-1, t, c, new_h, new_w)
                
                # Flip
                if flip_type == 'horizontal':
                    aug_value = torch.flip(aug_value, dims=[-1])
                elif flip_type == 'vertical':
                    aug_value = torch.flip(aug_value, dims=[-2])
                elif flip_type == 'both':
                    aug_value = torch.flip(aug_value, dims=[-2, -1])
                
                aug_batch[key] = aug_value
            else:
                aug_batch[key] = value
        
        return aug_batch
    
    def _reverse_tta_augmentation(
        self, 
        probs: torch.Tensor, 
        scale: float, 
        flip_type: str, 
        target_h: int, 
        target_w: int
    ) -> torch.Tensor:
        """Reverse TTA augmentation on predictions."""
        # Reverse flip
        if flip_type == 'horizontal':
            probs = torch.flip(probs, dims=[-1])
        elif flip_type == 'vertical':
            probs = torch.flip(probs, dims=[-2])
        elif flip_type == 'both':
            probs = torch.flip(probs, dims=[-2, -1])
        
        # Reverse scale
        if scale != 1.0 or probs.shape[-2:] != (target_h, target_w):
            probs = F.interpolate(
                probs, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        return probs
    
    def _apply_crf_postprocessing(
        self, 
        images: torch.Tensor, 
        probabilities: torch.Tensor, 
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Conditional Random Field post-processing.
        
        Args:
            images (torch.Tensor): Original images
            probabilities (torch.Tensor): Prediction probabilities
            predictions (torch.Tensor): Initial predictions
        
        Returns:
            torch.Tensor: CRF-refined predictions
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            logger.warning("pydensecrf not available. Skipping CRF post-processing.")
            return predictions
        
        refined_predictions = []
        
        for i in range(len(predictions)):
            # Convert to numpy
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            prob = probabilities[i].cpu().numpy()  # [C, H, W]
            
            # Normalize image to [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            h, w = img.shape[:2]
            
            # Setup CRF
            d = dcrf.DenseCRF2D(w, h, self.num_classes)
            
            # Unary potential
            unary = unary_from_softmax(prob)
            d.setUnaryEnergy(unary)
            
            # Pairwise potentials
            d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
            
            if img.shape[-1] >= 3:
                d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                                     compat=10, kernel=dcrf.DIAG_KERNEL,
                                     normalization=dcrf.NORMALIZE_SYMMETRIC)
            
            # Inference
            Q = d.inference(5)
            MAP = np.argmax(Q, axis=0).reshape((h, w))
            
            refined_predictions.append(torch.tensor(MAP, device=predictions.device))
        
        return torch.stack(refined_predictions)
    
    def _calculate_confidence(self, probabilities: torch.Tensor) -> float:
        """Calculate confidence score for prediction."""
        # Use entropy as uncertainty measure
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=0)
        confidence = 1.0 - (entropy / np.log(self.num_classes))  # Normalize entropy
        return confidence.mean().item()
    
    def _save_prediction_files(self, predictions: Dict[str, np.ndarray], output_dir: Path):
        """Save prediction files as .tif format."""
        logger.info(f"Saving prediction files to {output_dir}")
        
        for image_id, prediction in tqdm(predictions.items(), desc="Saving predictions"):
            output_file = output_dir / f"{image_id}_prediction.tif"
            
            # Save as GeoTIFF
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=prediction.dtype,
                compress='lzw'
            ) as dst:
                dst.write(prediction, 1)
    
    def _save_probability_files(self, probabilities: Dict[str, np.ndarray], output_dir: Path):
        """Save probability files as .tif format."""
        logger.info(f"Saving probability files to {output_dir}")
        
        for image_id, probability in tqdm(probabilities.items(), desc="Saving probabilities"):
            output_file = output_dir / f"{image_id}_probabilities.tif"
            
            num_classes = probability.shape[0]
            
            # Save as multi-band GeoTIFF
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=probability.shape[1],
                width=probability.shape[2],
                count=num_classes,
                dtype=np.float32,
                compress='lzw'
            ) as dst:
                for c in range(num_classes):
                    dst.write(probability[c].astype(np.float32), c + 1)
    
    def _generate_submission_csv(self, predictions: Dict[str, np.ndarray], output_file: Path):
        """Generate submission CSV file."""
        logger.info(f"Generating submission CSV: {output_file}")
        
        submission_data = []
        
        for image_id, prediction in predictions.items():
            # Create submission entry
            submission_data.append({
                'image_id': image_id,
                'prediction_file': f"{image_id}_prediction.tif"
            })
        
        # Create DataFrame and save
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_file, index=False)
        
        logger.info(f"Submission CSV saved with {len(submission_data)} entries")
    
    def _generate_submission_stats(
        self, 
        predictions: Dict[str, np.ndarray], 
        confidence_scores: List[float], 
        image_ids: List[str]
    ) -> Dict[str, Any]:
        """Generate submission statistics."""
        stats = {
            'total_images': len(predictions),
            'class_distribution': {},
            'confidence_stats': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            },
            'image_stats': {}
        }
        
        # Class distribution
        all_predictions = []
        for pred in predictions.values():
            all_predictions.extend(pred.flatten())
        
        unique, counts = np.unique(all_predictions, return_counts=True)
        total_pixels = len(all_predictions)
        
        for class_id, count in zip(unique, counts):
            if class_id < len(self.class_names):
                stats['class_distribution'][self.class_names[class_id]] = {
                    'count': int(count),
                    'percentage': float(count / total_pixels * 100)
                }
        
        # Per-image statistics
        for i, (image_id, pred) in enumerate(predictions.items()):
            unique_classes, class_counts = np.unique(pred, return_counts=True)
            dominant_class = unique_classes[np.argmax(class_counts)]
            
            stats['image_stats'][image_id] = {
                'dominant_class': self.class_names[dominant_class] if dominant_class < len(self.class_names) else 'unknown',
                'num_classes': len(unique_classes),
                'confidence': confidence_scores[i] if i < len(confidence_scores) else 0.0
            }
        
        return stats
    
    def _save_submission_stats(self, stats: Dict[str, Any], output_file: Path):
        """Save submission statistics."""
        import json
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Submission statistics saved to {output_file}")
    
    def validate_submission(self, submission_dir: str) -> Dict[str, Any]:
        """
        Validate submission format and files.
        
        Args:
            submission_dir (str): Directory containing submission files
        
        Returns:
            Dict[str, Any]: Validation results
        """
        submission_dir = Path(submission_dir)
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_count': 0,
            'format_check': True
        }
        
        # Check if submission CSV exists
        csv_files = list(submission_dir.glob('*.csv'))
        if not csv_files:
            validation_results['errors'].append("No submission CSV file found")
            validation_results['valid'] = False
        
        # Check prediction files
        pred_dir = submission_dir / 'predictions'
        if pred_dir.exists():
            tif_files = list(pred_dir.glob('*.tif'))
            validation_results['file_count'] = len(tif_files)
            
            # Sample validation
            for i, tif_file in enumerate(tif_files[:5]):  # Check first 5 files
                try:
                    with rasterio.open(tif_file) as src:
                        data = src.read(1)
                        
                        # Check data range
                        if data.min() < 0 or data.max() >= self.num_classes:
                            validation_results['warnings'].append(
                                f"File {tif_file.name} has invalid class values"
                            )
                        
                except Exception as e:
                    validation_results['errors'].append(
                        f"Cannot read file {tif_file.name}: {str(e)}"
                    )
                    validation_results['format_check'] = False
        
        if validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results


# Export for easy imports
__all__ = ['SubmissionGenerator']

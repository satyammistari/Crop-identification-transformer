"""
AgriFieldNetDataModule: PyTorch Lightning DataModule for the competition.

This module handles all data loading, preprocessing, and splitting for the
AMPT model training and evaluation. It provides standardized data loaders
with proper batching, shuffling, and multi-processing.

Features:
- Automatic train/val/test split management
- Multi-worker data loading
- Memory optimization
- Class weight computation
- Data statistics calculation
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional, Dict, Any, Tuple
import torch
from omegaconf import DictConfig
import logging

from .agrifieldnet_dataset import AgriFieldNetDataset

logger = logging.getLogger(__name__)


class AgriFieldNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AgriFieldNet dataset.
    
    This class handles all aspects of data management including:
    - Dataset initialization for train/val/test splits
    - Data loader configuration
    - Class balancing
    - Data statistics
    
    Args:
        config: Configuration object containing data parameters
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.batch_size = config.get('batch_size', 8)
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)
        self.persistent_workers = config.get('persistent_workers', True)
        self.prefetch_factor = config.get('prefetch_factor', 2)
        
        # Dataset parameters
        self.image_size = config.get('image_size', [256, 256])
        self.temporal_length = config.get('temporal_length', 6)
        self.include_sar = config.get('include_sar', True)
        self.include_weather = config.get('include_weather', True)
        self.cache_data = config.get('cache_data', False)
        
        # Augmentation and preprocessing
        self.augmentations = config.get('augmentations', {})
        self.normalize_stats = config.get('normalize_stats', {})
        self.synthetic_data = config.get('synthetic_data', {})
        
        # Memory optimization
        self.memory_opt = config.get('memory_optimization', {})
        
        # Class balancing
        self.use_weighted_sampling = config.get('use_weighted_sampling', False)
        
        # Initialize datasets to None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Store class weights
        self.class_weights = None
        
        logger.info(f"Initialized DataModule with batch_size={self.batch_size}, "
                   f"num_workers={self.num_workers}")
    
    def prepare_data(self):
        """
        Download or prepare data if needed.
        
        This method is called only once and on a single process.
        Use it for data download, extraction, etc.
        """
        # Check if data directory exists
        import os
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist. "
                          "Please ensure data is available.")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        
        Args:
            stage: 'fit' for train/val, 'test' for test, 'predict' for inference
        """
        if stage == 'fit' or stage is None:
            # Setup train dataset
            self.train_dataset = AgriFieldNetDataset(
                data_dir=self.data_dir,
                split='train',
                image_size=tuple(self.image_size),
                temporal_length=self.temporal_length,
                augmentations=self.augmentations,
                include_sar=self.include_sar,
                include_weather=self.include_weather,
                synthetic_data=self.synthetic_data,
                normalize_stats=self.normalize_stats,
                cache_data=self.cache_data and self.memory_opt.get('cache_data', False)
            )
            
            # Setup validation dataset
            self.val_dataset = AgriFieldNetDataset(
                data_dir=self.data_dir,
                split='val',
                image_size=tuple(self.image_size),
                temporal_length=self.temporal_length,
                augmentations=self.augmentations,
                include_sar=self.include_sar,
                include_weather=self.include_weather,
                synthetic_data=self.synthetic_data,
                normalize_stats=self.normalize_stats,
                cache_data=False  # Usually don't cache val data
            )
            
            # Compute class weights from training data
            if self.use_weighted_sampling:
                logger.info("Computing class weights for balanced sampling...")
                self.class_weights = self.train_dataset.get_class_weights()
                logger.info(f"Class weights: {self.class_weights}")
            
            logger.info(f"Setup train dataset: {len(self.train_dataset)} samples")
            logger.info(f"Setup val dataset: {len(self.val_dataset)} samples")
        
        if stage == 'test' or stage is None:
            # Setup test dataset
            self.test_dataset = AgriFieldNetDataset(
                data_dir=self.data_dir,
                split='test',
                image_size=tuple(self.image_size),
                temporal_length=self.temporal_length,
                augmentations={'test': {'enabled': False}},  # No augmentation for test
                include_sar=self.include_sar,
                include_weather=self.include_weather,
                synthetic_data=self.synthetic_data,
                normalize_stats=self.normalize_stats,
                cache_data=False
            )
            
            logger.info(f"Setup test dataset: {len(self.test_dataset)} samples")
        
        if stage == 'predict':
            # For prediction, use test dataset setup
            self.setup('test')
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        
        # Setup sampler for class balancing
        sampler = None
        shuffle = True
        
        if self.use_weighted_sampling and self.class_weights is not None:
            # Create weighted sampler for balanced training
            sample_weights = torch.zeros(len(self.train_dataset))
            
            for i in range(len(self.train_dataset)):
                sample = self.train_dataset[i]
                if 'labels' in sample:
                    labels = sample['labels']
                    valid_mask = sample['valid_mask']
                    
                    # Get dominant class in the sample
                    valid_labels = labels[valid_mask]
                    if len(valid_labels) > 0:
                        # Use mode (most frequent class) as sample weight
                        unique_labels, counts = torch.unique(valid_labels, return_counts=True)
                        dominant_class = unique_labels[torch.argmax(counts)]
                        sample_weights[i] = self.class_weights[dominant_class]
                    else:
                        sample_weights[i] = 1.0
                else:
                    sample_weights[i] = 1.0
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            shuffle = False  # Mutually exclusive with sampler
            
            logger.info("Using weighted sampling for balanced training")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
            drop_last=True,  # For stable batch norm
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Val dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
            drop_last=False,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
            drop_last=False,
            collate_fn=self._collate_fn
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader."""
        return self.test_dataloader()
    
    def _collate_fn(self, batch):
        """
        Custom collate function to handle variable data modalities.
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Dict: Batched data
        """
        # Initialize batch dictionary
        batched = {}
        
        # Get all keys from first sample
        sample_keys = batch[0].keys()
        
        for key in sample_keys:
            # Collect all values for this key
            values = [sample[key] for sample in batch if key in sample and sample[key] is not None]
            
            if len(values) == 0:
                batched[key] = None
            elif len(values) == len(batch):
                # All samples have this key
                if isinstance(values[0], torch.Tensor):
                    batched[key] = torch.stack(values, dim=0)
                elif isinstance(values[0], str):
                    batched[key] = values  # Keep as list for strings
                else:
                    batched[key] = torch.tensor(values)
            else:
                # Some samples missing this key - pad or handle appropriately
                if isinstance(values[0], torch.Tensor):
                    # Create tensor with None for missing values
                    full_values = []
                    for sample in batch:
                        if key in sample and sample[key] is not None:
                            full_values.append(sample[key])
                        else:
                            # Create zero tensor with same shape as first valid sample
                            zero_tensor = torch.zeros_like(values[0])
                            full_values.append(zero_tensor)
                    batched[key] = torch.stack(full_values, dim=0)
                else:
                    batched[key] = values
        
        return batched
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Compute and return dataset statistics.
        
        Returns:
            Dict containing various dataset statistics
        """
        stats = {
            'num_classes': 6,
            'class_names': ['gram', 'maize', 'mustard', 'sugarcane', 'wheat', 'other_crop'],
            'image_size': self.image_size,
            'temporal_length': self.temporal_length,
            'include_sar': self.include_sar,
            'include_weather': self.include_weather,
        }
        
        if self.train_dataset is not None:
            stats['train_samples'] = len(self.train_dataset)
        
        if self.val_dataset is not None:
            stats['val_samples'] = len(self.val_dataset)
        
        if self.test_dataset is not None:
            stats['test_samples'] = len(self.test_dataset)
        
        if self.class_weights is not None:
            stats['class_weights'] = self.class_weights.tolist()
        
        return stats
    
    def get_sample_data(self, split: str = 'train', index: int = 0) -> Dict[str, Any]:
        """
        Get a sample from the specified split for inspection.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            index: Sample index
        
        Returns:
            Dict: Sample data
        """
        if split == 'train' and self.train_dataset is not None:
            return self.train_dataset[index]
        elif split == 'val' and self.val_dataset is not None:
            return self.val_dataset[index]
        elif split == 'test' and self.test_dataset is not None:
            return self.test_dataset[index]
        else:
            raise ValueError(f"Dataset {split} not available or not initialized")
    
    def compute_normalization_stats(self, split: str = 'train', num_samples: int = 100) -> Dict[str, Dict[str, list]]:
        """
        Compute normalization statistics from a subset of the data.
        
        Args:
            split: Dataset split to use for computation
            num_samples: Number of samples to use
        
        Returns:
            Dict: Normalization statistics
        """
        if split == 'train' and self.train_dataset is not None:
            dataset = self.train_dataset
        elif split == 'val' and self.val_dataset is not None:
            dataset = self.val_dataset
        else:
            raise ValueError(f"Dataset {split} not available")
        
        # Collect samples
        optical_values = []
        sar_values = []
        weather_values = []
        
        num_samples = min(num_samples, len(dataset))
        indices = torch.randperm(len(dataset))[:num_samples]
        
        logger.info(f"Computing normalization stats from {num_samples} samples...")
        
        for idx in indices:
            sample = dataset[idx.item()]
            
            if 'optical' in sample:
                optical_values.append(sample['optical'].flatten())
            
            if 'sar' in sample and sample['sar'] is not None:
                sar_values.append(sample['sar'].flatten())
            
            if 'weather' in sample and sample['weather'] is not None:
                weather_values.append(sample['weather'].flatten())
        
        stats = {}
        
        # Optical statistics
        if optical_values:
            optical_tensor = torch.cat(optical_values)
            stats['optical'] = {
                'mean': optical_tensor.mean().item(),
                'std': optical_tensor.std().item(),
                'min': optical_tensor.min().item(),
                'max': optical_tensor.max().item()
            }
        
        # SAR statistics
        if sar_values:
            sar_tensor = torch.cat(sar_values)
            stats['sar'] = {
                'mean': sar_tensor.mean().item(),
                'std': sar_tensor.std().item(),
                'min': sar_tensor.min().item(),
                'max': sar_tensor.max().item()
            }
        
        # Weather statistics
        if weather_values:
            weather_tensor = torch.cat(weather_values)
            stats['weather'] = {
                'mean': weather_tensor.mean().item(),
                'std': weather_tensor.std().item(),
                'min': weather_tensor.min().item(),
                'max': weather_tensor.max().item()
            }
        
        logger.info("Normalization statistics computed successfully")
        
        return stats


# Export for easy imports
__all__ = ['AgriFieldNetDataModule']

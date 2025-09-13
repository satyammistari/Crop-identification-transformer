"""
Enhanced Data Module for AMPT Model Training

This module creates the proper data structure and sample data for training
the Enhanced AMPT model with comprehensive metrics.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import logging
from pathlib import Path

from .agrifieldnet_dataset import AgriFieldNetDataset, get_agrifieldnet_transforms

logger = logging.getLogger(__name__)

class EnhancedAgriFieldNetDataModule(pl.LightningDataModule):
    """
    Enhanced PyTorch Lightning DataModule for AgriFieldNet dataset.
    
    Supports the Enhanced AMPT model with:
    - Multi-modal data (Optical + SAR + Weather)
    - Temporal sequences (6 months)
    - Comprehensive data augmentation
    - Proper class balancing
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 224,
        temporal_length: int = 6,
        train_transform: bool = True,
        val_transform: bool = False
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.temporal_length = temporal_length
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        # Dataset configuration
        self.dataset_config = {
            'image_size': image_size,
            'temporal_length': temporal_length,
            'use_preprocessed': True
        }
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        logger.info(f"Enhanced AgriFieldNet DataModule initialized")
        logger.info(f"Data dir: {data_dir}, Batch size: {batch_size}")
    
    def prepare_data(self):
        """Prepare data (download, create samples if needed)."""
        data_path = Path(self.data_dir)
        
        # Check if data exists
        if not data_path.exists():
            logger.warning(f"Data directory {data_path} does not exist")
            logger.info("Creating sample data for demonstration...")
            self._create_sample_data()
        else:
            # Check for required splits
            required_splits = ['train', 'val', 'test']
            for split in required_splits:
                split_dir = data_path / split
                if not split_dir.exists():
                    logger.warning(f"Split directory {split_dir} not found")
                    self._create_sample_data()
                    break
    
    def _create_sample_data(self):
        """Create sample data for demonstration purposes."""
        import numpy as np
        import cv2
        import json
        
        logger.info("Creating sample AgriFieldNet data...")
        
        data_path = Path(self.data_dir)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            split_dir = data_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Number of samples per split
            num_samples = {'train': 10, 'val': 3, 'test': 3}[split]
            
            for i in range(num_samples):
                sample_id = f"agrifield_{i:06d}"
                
                # Create optical data (6 time steps, 224x224, 6 bands)
                optical_data = np.random.rand(6, 224, 224, 6).astype(np.float32)
                # Add some vegetation patterns
                for t in range(6):
                    # Simulate seasonal vegetation changes
                    veg_factor = 0.3 + 0.7 * np.sin(t * np.pi / 3)
                    optical_data[t, :, :, 3] *= veg_factor  # NIR band
                
                np.save(split_dir / f"{sample_id}_optical.npy", optical_data)
                
                # Create SAR data (6 time steps, 224x224, 2 polarizations)
                sar_data = np.random.randn(6, 224, 224, 2).astype(np.float32) * 0.1
                # Add some structure based on optical
                for t in range(6):
                    # VV polarization based on vegetation
                    sar_data[t, :, :, 0] = -0.1 + 0.2 * optical_data[t, :, :, 3]
                    # VH polarization
                    sar_data[t, :, :, 1] = -0.2 + 0.3 * optical_data[t, :, :, 3]
                
                np.save(split_dir / f"{sample_id}_sar.npy", sar_data)
                
                # Create weather data (5 features)
                weather_data = np.array([
                    25.0 + np.random.randn() * 5,  # Temperature
                    60.0 + np.random.randn() * 15,  # Humidity
                    10.0 + np.random.randn() * 5,   # Rainfall
                    15.0 + np.random.randn() * 3,   # Wind speed
                    1013.0 + np.random.randn() * 10  # Pressure
                ]).astype(np.float32)
                
                np.save(split_dir / f"{sample_id}_weather.npy", weather_data)
                
                # Create metadata
                metadata = {
                    'sample_id': sample_id,
                    'location': {'lat': 28.6 + np.random.randn() * 0.1, 
                               'lon': 77.2 + np.random.randn() * 0.1},
                    'date_range': ['2021-01-01', '2021-06-30'],
                    'crop_type': np.random.choice(['rice', 'wheat', 'sugarcane', 'cotton', 'maize', 'other'])
                }
                
                with open(split_dir / f"{sample_id}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create mask for train/val splits
                if split in ['train', 'val']:
                    # Create a simple mask with random crop classes
                    mask = np.random.randint(0, 6, (224, 224), dtype=np.uint8)
                    
                    # Add some spatial coherence
                    mask = cv2.medianBlur(mask, 5)
                    
                    # Save as PNG
                    cv2.imwrite(str(split_dir / f"{sample_id}_mask.png"), mask)
        
        # Create summary file
        summary = {
            'dataset_name': 'AgriFieldNet Sample',
            'num_classes': 6,
            'class_names': ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Other'],
            'image_size': [224, 224],
            'temporal_length': 6,
            'modalities': ['optical', 'sar', 'weather'],
            'splits': {
                'train': 10,
                'val': 3,
                'test': 3
            },
            'created': True
        }
        
        with open(data_path / 'agrifieldnet_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Sample data created in {data_path}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        
        # Get transforms
        train_transform = get_agrifieldnet_transforms('train', self.image_size) if self.train_transform else None
        val_transform = get_agrifieldnet_transforms('val', self.image_size) if self.val_transform else None
        
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = AgriFieldNetDataset(
                data_dir=self.data_dir,
                split='train',
                transform=train_transform,
                config=self.dataset_config
            )
            
            # Validation dataset
            self.val_dataset = AgriFieldNetDataset(
                data_dir=self.data_dir,
                split='val',
                transform=val_transform,
                config=self.dataset_config
            )
            
            logger.info(f"Training dataset: {len(self.train_dataset)} samples")
            logger.info(f"Validation dataset: {len(self.val_dataset)} samples")
        
        if stage == 'test' or stage is None:
            # Test dataset
            self.test_dataset = AgriFieldNetDataset(
                data_dir=self.data_dir,
                split='test',
                transform=val_transform,
                config=self.dataset_config
            )
            
            logger.info(f"Test dataset: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

# For backward compatibility
AgriFieldNetDataModule = EnhancedAgriFieldNetDataModule

__all__ = ['EnhancedAgriFieldNetDataModule', 'AgriFieldNetDataModule']

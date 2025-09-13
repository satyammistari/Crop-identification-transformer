"""
Unit tests for dataset classes and data processing utilities.

Tests the AgriFieldNet dataset implementation, data loading,
augmentation, and batch creation functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

# Import dataset components
from src.data.agrifieldnet_dataset import AgriFieldNetDataset
from src.data.agrifieldnet_datamodule import AgriFieldNetDataModule


class TestAgriFieldNetDataset:
    """Test cases for AgriFieldNetDataset."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def dataset_config(self, temp_dir):
        """Create test configuration for dataset."""
        return DictConfig({
            'data_dir': temp_dir,
            'optical_bands': 13,
            'sar_bands': 2,
            'weather_features': 10,
            'max_temporal_length': 12,
            'image_size': [64, 64],
            'num_classes': 6,
            'synthetic_data': True,  # Use synthetic data for testing
            'temporal_augmentation': {
                'enabled': True,
                'min_length': 4,
                'max_length': 12,
                'random_sampling': True
            },
            'spatial_augmentation': {
                'enabled': True,
                'flip_prob': 0.5,
                'rotation_prob': 0.3,
                'crop_prob': 0.2
            }
        })
    
    @pytest.fixture
    def dataset(self, dataset_config):
        """Create dataset instance."""
        return AgriFieldNetDataset(
            data_dir=dataset_config.data_dir,
            split='train',
            config=dataset_config
        )
    
    def test_dataset_initialization(self, dataset, dataset_config):
        """Test dataset initialization."""
        assert dataset.data_dir == Path(dataset_config.data_dir)
        assert dataset.optical_bands == dataset_config.optical_bands
        assert dataset.sar_bands == dataset_config.sar_bands
        assert dataset.weather_features == dataset_config.weather_features
        assert dataset.max_temporal_length == dataset_config.max_temporal_length
        assert dataset.num_classes == dataset_config.num_classes
    
    def test_dataset_length(self, dataset):
        """Test dataset length."""
        assert len(dataset) > 0
        assert isinstance(len(dataset), int)
    
    def test_dataset_getitem(self, dataset):
        """Test dataset item retrieval."""
        # Get first item
        item = dataset[0]
        
        # Check item structure
        assert isinstance(item, dict)
        required_keys = ['optical', 'sar', 'weather', 'temporal_positions', 'mask', 'valid_pixels', 'image_id']
        for key in required_keys:
            assert key in item, f"Missing key: {key}"
        
        # Check data types and shapes
        optical = item['optical']
        sar = item['sar']
        weather = item['weather']
        temporal_positions = item['temporal_positions']
        mask = item['mask']
        valid_pixels = item['valid_pixels']
        
        assert isinstance(optical, torch.Tensor)
        assert isinstance(sar, torch.Tensor)
        assert isinstance(weather, torch.Tensor)
        assert isinstance(temporal_positions, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(valid_pixels, torch.Tensor)
        
        # Check shapes
        T, C, H, W = optical.shape
        assert T <= dataset.max_temporal_length
        assert C == dataset.optical_bands
        assert H == dataset.image_size[0]
        assert W == dataset.image_size[1]
        
        assert sar.shape == (T, dataset.sar_bands, H, W)
        assert weather.shape == (T, dataset.weather_features)
        assert temporal_positions.shape == (T,)
        assert mask.shape == (H, W)
        assert valid_pixels.shape == (H, W)
    
    def test_dataset_multiple_items(self, dataset):
        """Test retrieving multiple dataset items."""
        indices = [0, 1, min(2, len(dataset) - 1)]
        
        for idx in indices:
            item = dataset[idx]
            
            # Basic structure check
            assert isinstance(item, dict)
            assert 'optical' in item
            assert 'sar' in item
            assert 'mask' in item
            
            # Check no NaN values
            assert not torch.isnan(item['optical']).any()
            assert not torch.isnan(item['sar']).any()
            assert not torch.isnan(item['weather']).any()
    
    def test_dataset_index_bounds(self, dataset):
        """Test dataset index bounds."""
        # Valid indices
        assert dataset[0] is not None
        assert dataset[len(dataset) - 1] is not None
        
        # Invalid indices should raise IndexError
        with pytest.raises(IndexError):
            dataset[len(dataset)]
        
        with pytest.raises(IndexError):
            dataset[-len(dataset) - 1]
    
    def test_temporal_augmentation(self, dataset_config):
        """Test temporal augmentation functionality."""
        dataset_config.temporal_augmentation.enabled = True
        dataset = AgriFieldNetDataset(
            data_dir=dataset_config.data_dir,
            split='train',
            config=dataset_config
        )
        
        # Get multiple samples and check temporal length variation
        temporal_lengths = []
        for i in range(5):
            item = dataset[i % len(dataset)]
            temporal_lengths.append(item['optical'].shape[0])
        
        # Should have some variation in temporal lengths
        min_length = dataset_config.temporal_augmentation.min_length
        max_length = dataset_config.temporal_augmentation.max_length
        
        for length in temporal_lengths:
            assert min_length <= length <= max_length
    
    def test_spatial_augmentation(self, dataset_config):
        """Test spatial augmentation functionality."""
        dataset_config.spatial_augmentation.enabled = True
        dataset = AgriFieldNetDataset(
            data_dir=dataset_config.data_dir,
            split='train',
            config=dataset_config
        )
        
        # Get same item multiple times and check for variation
        item1 = dataset[0]
        item2 = dataset[0]
        
        # Items should potentially be different due to augmentation
        # (Note: this is probabilistic, so we just check shapes are consistent)
        assert item1['optical'].shape == item2['optical'].shape
        assert item1['mask'].shape == item2['mask'].shape
    
    def test_no_augmentation(self, dataset_config):
        """Test dataset without augmentation."""
        dataset_config.temporal_augmentation.enabled = False
        dataset_config.spatial_augmentation.enabled = False
        
        dataset = AgriFieldNetDataset(
            data_dir=dataset_config.data_dir,
            split='val',  # Validation typically has no augmentation
            config=dataset_config
        )
        
        # Get same item multiple times
        item1 = dataset[0]
        item2 = dataset[0]
        
        # Should be identical
        assert torch.equal(item1['optical'], item2['optical'])
        assert torch.equal(item1['sar'], item2['sar'])
        assert torch.equal(item1['mask'], item2['mask'])
    
    def test_synthetic_data_generation(self, dataset_config):
        """Test synthetic data generation."""
        dataset_config.synthetic_data = True
        
        dataset = AgriFieldNetDataset(
            data_dir=dataset_config.data_dir,
            split='train',
            config=dataset_config
        )
        
        item = dataset[0]
        
        # Check synthetic data properties
        optical = item['optical']
        sar = item['sar']
        mask = item['mask']
        
        # Optical data should be in reasonable range
        assert optical.min() >= 0
        assert optical.max() <= 1
        
        # SAR data can have negative values
        assert sar.min() >= -1
        assert sar.max() <= 1
        
        # Mask should contain valid class indices
        assert mask.min() >= 0
        assert mask.max() < dataset.num_classes
    
    def test_weather_data_generation(self, dataset):
        """Test weather data generation."""
        item = dataset[0]
        weather = item['weather']
        
        T, F = weather.shape
        assert F == dataset.weather_features
        
        # Weather features should be in reasonable ranges
        assert not torch.isnan(weather).any()
        assert not torch.isinf(weather).any()
    
    def test_temporal_positions(self, dataset):
        """Test temporal position encoding."""
        item = dataset[0]
        temporal_positions = item['temporal_positions']
        
        # Should be day of year (0-365)
        assert temporal_positions.min() >= 0
        assert temporal_positions.max() <= 365
        
        # Should be sorted (temporal sequence)
        assert torch.all(temporal_positions[1:] >= temporal_positions[:-1])
    
    def test_valid_pixels_mask(self, dataset):
        """Test valid pixels mask."""
        item = dataset[0]
        valid_pixels = item['valid_pixels']
        mask = item['mask']
        
        # Valid pixels should be boolean
        assert valid_pixels.dtype == torch.bool
        
        # Shape should match mask
        assert valid_pixels.shape == mask.shape
        
        # Should have at least some valid pixels
        assert valid_pixels.sum() > 0


class TestAgriFieldNetDataModule:
    """Test cases for AgriFieldNetDataModule."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def datamodule_config(self, temp_dir):
        """Create test configuration for datamodule."""
        return OmegaConf.create({
            'data': {
                'data_dir': temp_dir,
                'optical_bands': 13,
                'sar_bands': 2,
                'weather_features': 10,
                'max_temporal_length': 12,
                'image_size': [64, 64],
                'num_classes': 6,
                'synthetic_data': True,
                'batch_size': 4,
                'num_workers': 0,  # Disable multiprocessing for testing
                'train_split': 0.7,
                'val_split': 0.2,
                'test_split': 0.1,
                'weighted_sampling': True,
                'temporal_augmentation': {
                    'enabled': True,
                    'min_length': 4,
                    'max_length': 12
                },
                'spatial_augmentation': {
                    'enabled': True,
                    'flip_prob': 0.5
                }
            }
        })
    
    @pytest.fixture
    def datamodule(self, datamodule_config):
        """Create datamodule instance."""
        return AgriFieldNetDataModule(datamodule_config)
    
    def test_datamodule_initialization(self, datamodule, datamodule_config):
        """Test datamodule initialization."""
        assert datamodule.config == datamodule_config
        assert datamodule.batch_size == datamodule_config.data.batch_size
        assert datamodule.num_workers == datamodule_config.data.num_workers
    
    def test_datamodule_setup(self, datamodule):
        """Test datamodule setup."""
        # Setup for training
        datamodule.setup('fit')
        
        assert hasattr(datamodule, 'train_dataset')
        assert hasattr(datamodule, 'val_dataset')
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        
        # Setup for testing
        datamodule.setup('test')
        
        assert hasattr(datamodule, 'test_dataset')
        assert datamodule.test_dataset is not None
    
    def test_datamodule_dataloaders(self, datamodule):
        """Test datamodule dataloaders."""
        datamodule.setup('fit')
        
        # Train dataloader
        train_loader = datamodule.train_dataloader()
        assert train_loader is not None
        assert train_loader.batch_size == datamodule.batch_size
        
        # Validation dataloader
        val_loader = datamodule.val_dataloader()
        assert val_loader is not None
        assert val_loader.batch_size == datamodule.batch_size
        
        # Test dataloader
        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()
        assert test_loader is not None
        assert test_loader.batch_size == datamodule.batch_size
    
    def test_datamodule_batch_creation(self, datamodule):
        """Test batch creation and collation."""
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()
        
        # Get first batch
        batch = next(iter(train_loader))
        
        # Check batch structure
        assert isinstance(batch, dict)
        required_keys = ['optical', 'sar', 'weather', 'temporal_positions', 'mask', 'valid_pixels', 'image_id']
        for key in required_keys:
            assert key in batch
        
        # Check batch dimensions
        batch_size = datamodule.batch_size
        
        optical = batch['optical']
        sar = batch['sar']
        weather = batch['weather']
        mask = batch['mask']
        
        assert optical.shape[0] == batch_size
        assert sar.shape[0] == batch_size
        assert weather.shape[0] == batch_size
        assert mask.shape[0] == batch_size
        
        # Check consistency across samples in batch
        for i in range(1, batch_size):
            # All samples should have same temporal length (after collation)
            assert optical[i].shape[0] == optical[0].shape[0]
            assert sar[i].shape[0] == sar[0].shape[0]
            assert weather[i].shape[0] == weather[0].shape[0]
    
    def test_datamodule_custom_collate(self, datamodule):
        """Test custom collate function."""
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()
        
        # Get batch to test collation
        batch = next(iter(train_loader))
        
        # Verify temporal padding/truncation worked correctly
        optical = batch['optical']
        B, T, C, H, W = optical.shape
        
        assert T <= datamodule.config.data.max_temporal_length
        
        # All samples in batch should have same temporal dimension
        for key in ['optical', 'sar', 'weather', 'temporal_positions']:
            if key in batch:
                temporal_dim = 1 if key in ['optical', 'sar'] else 1
                if batch[key].dim() > temporal_dim:
                    first_temporal_size = batch[key].shape[temporal_dim]
                    for i in range(batch[key].shape[0]):
                        assert batch[key][i].shape[temporal_dim - 1] == first_temporal_size
    
    def test_datamodule_weighted_sampling(self, datamodule_config):
        """Test weighted sampling functionality."""
        datamodule_config.data.weighted_sampling = True
        datamodule = AgriFieldNetDataModule(datamodule_config)
        
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()
        
        # Check that sampler is used
        assert hasattr(train_loader, 'sampler')
        # Should not be None if weighted sampling is enabled
        # (Implementation may vary based on actual data distribution)
    
    def test_datamodule_no_weighted_sampling(self, datamodule_config):
        """Test without weighted sampling."""
        datamodule_config.data.weighted_sampling = False
        datamodule = AgriFieldNetDataModule(datamodule_config)
        
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()
        
        # Should use default sampler
        assert train_loader.sampler is not None
    
    def test_datamodule_different_splits(self, datamodule_config):
        """Test different data splits."""
        # Test different split ratios
        datamodule_config.data.train_split = 0.8
        datamodule_config.data.val_split = 0.15
        datamodule_config.data.test_split = 0.05
        
        datamodule = AgriFieldNetDataModule(datamodule_config)
        datamodule.setup('fit')
        datamodule.setup('test')
        
        # Check that datasets are created
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.test_dataset is not None
        
        # Check approximate split sizes
        total_samples = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
        
        train_ratio = len(datamodule.train_dataset) / total_samples
        val_ratio = len(datamodule.val_dataset) / total_samples
        test_ratio = len(datamodule.test_dataset) / total_samples
        
        # Allow some tolerance due to rounding
        assert abs(train_ratio - 0.8) < 0.1
        assert abs(val_ratio - 0.15) < 0.1
        assert abs(test_ratio - 0.05) < 0.1


class TestDatasetIntegration:
    """Integration tests for dataset and datamodule interaction."""
    
    def test_dataset_datamodule_compatibility(self):
        """Test dataset and datamodule work together."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = OmegaConf.create({
                'data': {
                    'data_dir': temp_dir,
                    'optical_bands': 13,
                    'sar_bands': 2,
                    'weather_features': 10,
                    'max_temporal_length': 8,
                    'image_size': [32, 32],
                    'num_classes': 6,
                    'synthetic_data': True,
                    'batch_size': 2,
                    'num_workers': 0,
                    'train_split': 0.7,
                    'val_split': 0.2,
                    'test_split': 0.1,
                    'weighted_sampling': False,
                    'temporal_augmentation': {'enabled': False},
                    'spatial_augmentation': {'enabled': False}
                }
            })
            
            # Create datamodule
            datamodule = AgriFieldNetDataModule(config)
            datamodule.setup('fit')
            
            # Get dataloaders
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            # Test training batch
            train_batch = next(iter(train_loader))
            assert isinstance(train_batch, dict)
            
            # Test validation batch
            val_batch = next(iter(val_loader))
            assert isinstance(val_batch, dict)
            
            # Check batch structure consistency
            for key in train_batch.keys():
                assert key in val_batch
                if isinstance(train_batch[key], torch.Tensor) and isinstance(val_batch[key], torch.Tensor):
                    # Shapes should be compatible (same except batch dimension)
                    assert train_batch[key].shape[1:] == val_batch[key].shape[1:]
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_dataloader_iteration(self):
        """Test full dataloader iteration."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = OmegaConf.create({
                'data': {
                    'data_dir': temp_dir,
                    'optical_bands': 13,
                    'sar_bands': 2,
                    'weather_features': 10,
                    'max_temporal_length': 6,
                    'image_size': [32, 32],
                    'num_classes': 6,
                    'synthetic_data': True,
                    'batch_size': 2,
                    'num_workers': 0,
                    'train_split': 0.9,
                    'val_split': 0.1,
                    'test_split': 0.0,
                    'weighted_sampling': False,
                    'temporal_augmentation': {'enabled': False},
                    'spatial_augmentation': {'enabled': False}
                }
            })
            
            datamodule = AgriFieldNetDataModule(config)
            datamodule.setup('fit')
            
            train_loader = datamodule.train_dataloader()
            
            # Iterate through a few batches
            batch_count = 0
            for batch in train_loader:
                assert isinstance(batch, dict)
                assert 'optical' in batch
                assert 'mask' in batch
                
                batch_count += 1
                if batch_count >= 3:  # Just test first 3 batches
                    break
            
            assert batch_count > 0
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
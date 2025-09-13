"""
AgriFieldNet Dataset Loader for AMPT Model
Handles real Sentinel-2 satellite imagery from the AgriFieldNet competition.
Compatible with both downloaded MLHub data and synthetic preprocessed data.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
import logging

logger = logging.getLogger(__name__)

class AgriFieldNetDataset(Dataset):
    """Dataset loader for AgriFieldNet satellite imagery with multi-modal support."""
    
    def __init__(self, data_dir, split='train', transform=None, config=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.config = config or {}
        
        # Dataset configuration
        self.image_size = self.config.get('image_size', 256)
        self.temporal_length = self.config.get('temporal_length', 6)
        self.use_preprocessed = self.config.get('use_preprocessed', True)
        
        # Load dataset summary if available
        summary_path = self.data_dir / 'agrifieldnet_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}
        
        # Class information
        self.num_classes = 6
        self.class_names = ['gram', 'maize', 'mustard', 'sugarcane', 'wheat', 'other_crop']
        
        # Load samples based on data format
        if self.use_preprocessed:
            self.samples = self._load_preprocessed_samples()
        else:
            self.samples = self._load_raw_agrifieldnet_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        if self.use_preprocessed:
            logger.info("Using preprocessed multi-modal data")
        else:
            logger.info("Using raw AgriFieldNet .tif files")
    
    def _load_preprocessed_samples(self):
        """Load preprocessed multi-modal samples."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} not found")
            return samples
        
        # Find all optical files and derive other file paths
        optical_files = list(split_dir.glob('*_optical.npy'))
        
        for optical_file in optical_files:
            sample_id = optical_file.stem.replace('_optical', '')
            
            # Define all required files
            sample_files = {
                'optical': optical_file,
                'sar': split_dir / f"{sample_id}_sar.npy",
                'weather': split_dir / f"{sample_id}_weather.npy",
                'metadata': split_dir / f"{sample_id}_metadata.json"
            }
            
            # Add mask for train/val splits
            if self.split in ['train', 'val']:
                sample_files['mask'] = split_dir / f"{sample_id}_mask.png"
            
            # Check if all required files exist
            required_files = ['optical', 'sar', 'weather']
            if self.split in ['train', 'val']:
                required_files.append('mask')
            
            if all(sample_files[key].exists() for key in required_files):
                samples.append(sample_files)
            else:
                missing = [key for key in required_files if not sample_files[key].exists()]
                logger.debug(f"Missing files for {sample_id}: {missing}")
        
        return samples
    
    def _load_raw_agrifieldnet_samples(self):
        """Load raw AgriFieldNet .tif files."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} not found")
            return samples
        
        # Find all .tif image files
        image_files = sorted(list(split_dir.glob("*.tif")))
        
        # Filter out label files
        image_files = [f for f in image_files 
                      if not any(suffix in f.stem.lower() 
                               for suffix in ['label', 'mask', 'ground_truth', 'gt'])]
        
        for img_file in image_files:
            # Try to find corresponding label file
            possible_labels = [
                split_dir / f"{img_file.stem}_label.tif",
                split_dir / f"{img_file.stem}_mask.tif", 
                split_dir / f"{img_file.stem}_gt.tif",
                split_dir / f"label_{img_file.name}",
                split_dir / f"mask_{img_file.name}",
            ]
            
            label_file = None
            for possible_label in possible_labels:
                if possible_label.exists():
                    label_file = possible_label
                    break
            
            # For test split, labels might not exist
            if self.split == 'test' and label_file is None:
                label_file = None
            
            samples.append({
                'image': img_file,
                'label': label_file
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_preprocessed:
            return self._get_preprocessed_item(idx)
        else:
            return self._get_raw_item(idx)
    
    def _get_preprocessed_item(self, idx):
        """Get item from preprocessed multi-modal data."""
        sample_files = self.samples[idx]
        
        try:
            # Load optical data: (T, H, W, C)
            optical_data = np.load(sample_files['optical'])  # Shape: (6, H, W, 6)
            
            # Load SAR data: (T, H, W, C)
            sar_data = np.load(sample_files['sar'])  # Shape: (6, H, W, 2)
            
            # Load weather data: (5,)
            weather_data = np.load(sample_files['weather'])  # Shape: (5,)
            
            # Load metadata
            if sample_files['metadata'].exists():
                with open(sample_files['metadata'], 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Process optical data (take middle time step for static processing)
            middle_step = optical_data.shape[0] // 2
            optical_img = optical_data[middle_step]  # Shape: (H, W, 6)
            
            # Process SAR data (take middle time step)
            sar_img = sar_data[middle_step]  # Shape: (H, W, 2)
            
            # Load mask (if available)
            mask = None
            if 'mask' in sample_files and sample_files['mask'].exists():
                mask = cv2.imread(str(sample_files['mask']), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Ensure mask values are in correct range
                    mask = np.clip(mask, 0, self.num_classes - 1)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(
                    image=optical_img,
                    mask=mask if mask is not None else np.zeros(optical_img.shape[:2], dtype=np.uint8)
                )
                optical_img = transformed['image']
                if mask is not None:
                    mask = transformed['mask']
            
            # Convert to tensors
            # Optical: (C, H, W) with 6 spectral bands
            if isinstance(optical_img, np.ndarray):
                optical_img = torch.from_numpy(optical_img).permute(2, 0, 1).float()
            
            # SAR: (C, H, W) with 2 polarizations
            sar_img = torch.from_numpy(sar_img).permute(2, 0, 1).float()
            
            # Weather: (5,)
            weather_data = torch.from_numpy(weather_data).float()
            
            # Temporal optical data: (T, C, H, W)
            temporal_optical = torch.from_numpy(optical_data).permute(0, 3, 1, 2).float()
            
            # Temporal SAR data: (T, C, H, W)
            temporal_sar = torch.from_numpy(sar_data).permute(0, 3, 1, 2).float()
            
            # Prepare output
            sample = {
                'optical': optical_img,  # Current time step: (6, H, W)
                'sar': sar_img,  # Current time step: (2, H, W)
                'weather': weather_data,  # (5,)
                'temporal_optical': temporal_optical,  # (6, 6, H, W)
                'temporal_sar': temporal_sar,  # (6, 2, H, W)
                'metadata': metadata,
                'sample_id': metadata.get('sample_id', f'sample_{idx}')
            }
            
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).long()
                sample['mask'] = mask
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading preprocessed sample {idx}: {e}")
            # Return a dummy sample
            h, w = 256, 256
            return {
                'optical': torch.zeros(6, h, w),
                'sar': torch.zeros(2, h, w),
                'weather': torch.zeros(5),
                'temporal_optical': torch.zeros(6, 6, h, w),
                'temporal_sar': torch.zeros(6, 2, h, w),
                'mask': torch.zeros(h, w, dtype=torch.long),
                'metadata': {},
                'sample_id': f'dummy_{idx}'
            }
    
    def _get_raw_item(self, idx):
        """Get item from raw AgriFieldNet .tif files."""
        sample = self.samples[idx]
        
        try:
            # Load satellite image
            image = self._load_tif_image(sample['image'])
            
            # Load label if available
            label = None
            if sample['label'] is not None:
                label = self._load_tif_label(sample['label'])
            
            # Generate synthetic multi-modal data
            optical_temporal = self._generate_temporal_optical(image)
            sar_temporal = self._generate_synthetic_sar(image)
            weather_data = self._generate_synthetic_weather()
            
            # Process current time step
            middle_step = optical_temporal.shape[0] // 2
            optical_img = optical_temporal[middle_step]
            sar_img = sar_temporal[middle_step]
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(
                    image=optical_img,
                    mask=label if label is not None else np.zeros(optical_img.shape[:2], dtype=np.uint8)
                )
                optical_img = transformed['image']
                if label is not None:
                    label = transformed['mask']
            
            # Convert to tensors
            if isinstance(optical_img, np.ndarray):
                optical_img = torch.from_numpy(optical_img).permute(2, 0, 1).float()
            
            sar_img = torch.from_numpy(sar_img).permute(2, 0, 1).float()
            weather_data = torch.from_numpy(weather_data).float()
            optical_temporal = torch.from_numpy(optical_temporal).permute(0, 3, 1, 2).float()
            sar_temporal = torch.from_numpy(sar_temporal).permute(0, 3, 1, 2).float()
            
            sample_dict = {
                'optical': optical_img,
                'sar': sar_img,
                'weather': weather_data,
                'temporal_optical': optical_temporal,
                'temporal_sar': sar_temporal,
                'metadata': {'image_path': str(sample['image'])},
                'sample_id': sample['image'].stem
            }
            
            if label is not None:
                if isinstance(label, np.ndarray):
                    label = torch.from_numpy(label).long()
                sample_dict['mask'] = label
            
            return sample_dict
            
        except Exception as e:
            logger.error(f"Error loading raw sample {idx}: {e}")
            h, w = 256, 256
            return {
                'optical': torch.zeros(6, h, w),
                'sar': torch.zeros(2, h, w),
                'weather': torch.zeros(5),
                'temporal_optical': torch.zeros(6, 6, h, w),
                'temporal_sar': torch.zeros(6, 2, h, w),
                'mask': torch.zeros(h, w, dtype=torch.long),
                'metadata': {},
                'sample_id': f'raw_dummy_{idx}'
            }
    
    def _load_tif_image(self, image_path):
        """Load .tif satellite image."""
        try:
            with rasterio.open(image_path) as src:
                image = src.read()  # Shape: (bands, height, width)
                image = np.transpose(image, (1, 2, 0))  # (H, W, C)
                
                # Handle different band configurations
                if image.shape[-1] == 3:  # RGB
                    # Expand to 6 bands
                    nir_approx = image[:, :, 1:2]  # Green as NIR
                    swir_approx = image[:, :, 0:2]  # R,G as SWIR
                    image = np.concatenate([image, nir_approx, swir_approx], axis=-1)
                elif image.shape[-1] >= 6:
                    image = image[:, :, :6]
                else:
                    # Pad to 6 channels
                    pad_width = ((0, 0), (0, 0), (0, 6 - image.shape[-1]))
                    image = np.pad(image, pad_width, mode='edge')
                
                # Normalize to 0-1 range
                if image.dtype == np.uint16:
                    image = image.astype(np.float32) / 65535.0
                elif image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                else:
                    image = image.astype(np.float32)
                    image = np.clip(image, 0, 1)
                
                return image
                
        except Exception as e:
            logger.error(f"Error loading TIF image {image_path}: {e}")
            return np.zeros((self.image_size, self.image_size, 6), dtype=np.float32)
    
    def _load_tif_label(self, label_path):
        """Load .tif label file."""
        try:
            with rasterio.open(label_path) as src:
                label = src.read(1)  # Read first band
                
                # Map AgriFieldNet classes to AMPT classes
                agrifieldnet_to_ampt = {
                    1: 4,    # Wheat -> wheat
                    2: 2,    # Mustard -> mustard
                    3: 5,    # Lentil -> other_crop
                    4: 5,    # No Crop -> other_crop
                    5: 5,    # Green pea -> other_crop
                    6: 3,    # Sugarcane -> sugarcane
                    8: 5,    # Garlic -> other_crop
                    9: 1,    # Maize -> maize
                    13: 0,   # Gram -> gram
                    14: 5,   # Coriander -> other_crop
                    15: 5,   # Potato -> other_crop
                    16: 5,   # Bersem -> other_crop
                    36: 5    # Rice -> other_crop (could map to grain)
                }
                
                # Create new label array
                new_label = np.full_like(label, 5, dtype=np.uint8)  # Default to other_crop
                
                for agri_class, ampt_class in agrifieldnet_to_ampt.items():
                    new_label[label == agri_class] = ampt_class
                
                return new_label
                
        except Exception as e:
            logger.error(f"Error loading TIF label {label_path}: {e}")
            return None
    
    def _generate_temporal_optical(self, base_image):
        """Generate temporal sequence from base optical image."""
        temporal_sequence = []
        
        for t in range(self.temporal_length):
            # Create seasonal variation
            season_factor = t / (self.temporal_length - 1)
            
            # Add variation to vegetation bands (NIR especially)
            image_t = base_image.copy()
            
            if image_t.shape[-1] >= 4:  # Has NIR
                # Simulate vegetation growth
                growth_factor = 0.8 + 0.4 * np.sin(season_factor * 2 * np.pi)
                image_t[:, :, 3] *= growth_factor  # NIR band
            
            # Add temporal noise
            noise = np.random.normal(0, 0.02, image_t.shape)
            image_t = np.clip(image_t + noise, 0, 1)
            
            temporal_sequence.append(image_t)
        
        return np.array(temporal_sequence)
    
    def _generate_synthetic_sar(self, optical_image):
        """Generate synthetic SAR data from optical image."""
        h, w = optical_image.shape[:2]
        sar_sequence = []
        
        for t in range(self.temporal_length):
            # Use vegetation information to simulate SAR backscatter
            if optical_image.shape[-1] >= 4:
                nir = optical_image[:, :, 3]
                red = optical_image[:, :, 0]
                
                # NDVI-based SAR simulation
                ndvi = (nir - red) / (nir + red + 1e-8)
                
                # VV polarization
                vv = -0.1 + 0.3 * np.clip(ndvi + 1, 0, 1)
                
                # VH polarization  
                vh = -0.2 + 0.4 * np.clip(ndvi + 1, 0, 1)
            else:
                # Fallback
                gray = np.mean(optical_image, axis=-1)
                vv = -0.1 + 0.2 * gray
                vh = -0.2 + 0.3 * gray
            
            # Add SAR speckle noise
            vv += np.random.normal(0, 0.05, vv.shape)
            vh += np.random.normal(0, 0.05, vh.shape)
            
            vv = np.clip(vv, -1, 1)
            vh = np.clip(vh, -1, 1)
            
            sar_t = np.stack([vv, vh], axis=-1)
            sar_sequence.append(sar_t)
        
        return np.array(sar_sequence)
    
    def _generate_synthetic_weather(self):
        """Generate synthetic weather time series."""
        weather_sequence = []
        
        for t in range(self.temporal_length):
            # Seasonal patterns
            season_phase = t / (self.temporal_length - 1) * 2 * np.pi
            
            temp = 25 + 10 * np.sin(season_phase)  # Temperature
            humidity = 65 + 15 * np.sin(season_phase + np.pi/4)  # Humidity
            rainfall = 5 + 8 * np.abs(np.sin(season_phase))  # Rainfall
            wind = 12 + 5 * np.random.random()  # Wind speed
            pressure = 1013 + 10 * np.sin(season_phase)  # Pressure
            
            weather_t = np.array([temp, humidity, rainfall, wind, pressure])
            weather_sequence.append(weather_t)
        
        return np.array(weather_sequence)

def get_agrifieldnet_transforms(split='train', image_size=256):
    """Get data transforms for AgriFieldNet dataset."""
    
    if split == 'train':
        # Training transforms with augmentation
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=10,
                    val_shift_limit=10,
                    p=1.0
                )
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],  # 6-band normalization
                std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
                std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])
    
    return transform

def create_agrifieldnet_dataloaders(data_dir, config):
    """Create data loaders for AgriFieldNet dataset."""
    
    # Get transforms
    train_transform = get_agrifieldnet_transforms('train', config.get('image_size', 256))
    val_transform = get_agrifieldnet_transforms('val', config.get('image_size', 256))
    
    # Create datasets
    train_dataset = AgriFieldNetDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        config=config
    )
    
    val_dataset = AgriFieldNetDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        config=config
    )
    
    test_dataset = AgriFieldNetDataset(
        data_dir=data_dir,
        split='test',
        transform=val_transform,
        config=config
    )
    
    # Create data loaders
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def visualize_agrifieldnet_sample(dataset, idx=0, save_path=None):
    """Visualize a sample from the AgriFieldNet dataset."""
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'AgriFieldNet Sample: {sample["sample_id"]}', fontsize=16)
    
    # Convert tensors to numpy for visualization
    optical = sample['optical'].permute(1, 2, 0).cpu().numpy()
    sar = sample['sar'].permute(1, 2, 0).cpu().numpy()
    weather = sample['weather'].cpu().numpy()
    
    # RGB composite (B04, B03, B02) - Red, Green, Blue
    rgb_indices = [2, 1, 0]  # B04, B03, B02 in our band order
    rgb_img = optical[:, :, rgb_indices]
    rgb_img = np.clip(rgb_img, 0, 1)
    
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB Composite (B04-B03-B02)')
    axes[0, 0].axis('off')
    
    # False color (NIR, Red, Green)
    nir_idx = 3  # B08 (NIR)
    false_color = optical[:, :, [nir_idx, 2, 1]]  # NIR, Red, Green
    false_color = np.clip(false_color, 0, 1)
    
    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title('False Color (B08-B04-B03)')
    axes[0, 1].axis('off')
    
    # SWIR composite
    swir1_idx, swir2_idx = 4, 5  # B11, B12
    swir_composite = optical[:, :, [swir2_idx, swir1_idx, 2]]  # SWIR2, SWIR1, Red
    swir_composite = np.clip(swir_composite, 0, 1)
    
    axes[0, 2].imshow(swir_composite)
    axes[0, 2].set_title('SWIR Composite (B12-B11-B04)')
    axes[0, 2].axis('off')
    
    # SAR VV polarization
    sar_vv = sar[:, :, 0]
    im1 = axes[1, 0].imshow(sar_vv, cmap='gray')
    axes[1, 0].set_title('SAR VV Polarization')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    # SAR VH polarization
    sar_vh = sar[:, :, 1]
    im2 = axes[1, 1].imshow(sar_vh, cmap='gray')
    axes[1, 1].set_title('SAR VH Polarization')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # Weather data
    weather_labels = ['Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Wind Speed (km/h)', 'Pressure (hPa)']
    axes[1, 2].bar(range(len(weather)), weather)
    axes[1, 2].set_title('Weather Data')
    axes[1, 2].set_xticks(range(len(weather)))
    axes[1, 2].set_xticklabels(weather_labels, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Show mask if available
    if 'mask' in sample:
        mask = sample['mask'].cpu().numpy()
        
        # Create a new figure for the mask
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title('RGB Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        class_colors = plt.cm.tab10(np.linspace(0, 1, dataset.num_classes))
        plt.imshow(mask, cmap='tab10', vmin=0, vmax=dataset.num_classes-1)
        plt.title('Crop Segmentation Mask')
        plt.axis('off')
        
        # Add colorbar with class names
        cbar = plt.colorbar(fraction=0.046)
        cbar.set_ticks(range(dataset.num_classes))
        cbar.set_ticklabels(dataset.class_names)
        
        if save_path:
            mask_path = save_path.replace('.png', '_mask.png')
            plt.savefig(mask_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    # Print sample information
    print(f"\nSample Information:")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Optical shape: {sample['optical'].shape}")
    print(f"SAR shape: {sample['sar'].shape}")
    print(f"Weather shape: {sample['weather'].shape}")
    print(f"Temporal optical shape: {sample['temporal_optical'].shape}")
    print(f"Temporal SAR shape: {sample['temporal_sar'].shape}")
    if 'mask' in sample:
        print(f"Mask shape: {sample['mask'].shape}")
        unique_classes = torch.unique(sample['mask'])
        print(f"Classes present: {unique_classes.tolist()}")
    print(f"Weather data: {sample['weather'].tolist()}")

def main():
    """Test the AgriFieldNet dataset loader."""
    # Test dataset loading
    data_dir = "data"
    config = {
        'batch_size': 2,
        'num_workers': 0,  # For testing
        'image_size': 256,
        'use_preprocessed': True  # Try preprocessed data first
    }
    
    # Create datasets
    train_dataset = AgriFieldNetDataset(data_dir, 'train', config=config)
    
    if len(train_dataset) > 0:
        print(f"Dataset loaded successfully with {len(train_dataset)} samples")
        
        # Visualize first sample
        visualize_agrifieldnet_sample(train_dataset, 0, 'agrifieldnet_sample_visualization.png')
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_agrifieldnet_dataloaders(data_dir, config)
        
        # Test batch loading
        for batch in train_loader:
            print(f"\nBatch loaded successfully:")
            print(f"Optical batch shape: {batch['optical'].shape}")
            print(f"SAR batch shape: {batch['sar'].shape}")
            print(f"Weather batch shape: {batch['weather'].shape}")
            if 'mask' in batch:
                print(f"Mask batch shape: {batch['mask'].shape}")
            break
    else:
        print("No samples found. Please run the data download script first:")
        print("python scripts/download_agrifieldnet.py")

if __name__ == "__main__":
    main()

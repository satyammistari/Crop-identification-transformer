"""
Preprocessing script for Crops_data.csv to create synthetic satellite-like data
for demonstration purposes with the AMPT model.
"""

import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json

def create_synthetic_satellite_images(df, output_dir, image_size=(256, 256)):
    """
    Create synthetic satellite-like images from agricultural statistics data.
    This simulates multi-temporal, multi-spectral satellite imagery.
    """
    
    # Create output directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val" 
    test_dir = Path(output_dir) / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Crop categories mapping (simplified from 6 classes in config)
    crop_mapping = {
        'RICE': 0,      # gram -> rice
        'WHEAT': 1,     # maize -> wheat  
        'MAIZE': 2,     # mustard -> maize
        'SUGARCANE': 3, # sugarcane
        'COTTON': 4,    # wheat -> cotton
        'OTHER': 5      # other_crop
    }
    
    # Select key crop features for synthetic image generation
    crop_features = [
        'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'RICE YIELD (Kg per ha)',
        'WHEAT AREA (1000 ha)', 'WHEAT PRODUCTION (1000 tons)', 'WHEAT YIELD (Kg per ha)',
        'MAIZE AREA (1000 ha)', 'MAIZE PRODUCTION (1000 tons)', 'MAIZE YIELD (Kg per ha)',
        'SUGARCANE AREA (1000 ha)', 'SUGARCANE PRODUCTION (1000 tons)', 'SUGARCANE YIELD (Kg per ha)',
        'COTTON AREA (1000 ha)', 'COTTON PRODUCTION (1000 tons)', 'COTTON YIELD (Kg per ha)'
    ]
    
    # Fill NaN values
    df[crop_features] = df[crop_features].fillna(0)
    
    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(df[crop_features])
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    datasets = {
        'train': (train_df, train_dir),
        'val': (val_df, val_dir), 
        'test': (test_df, test_dir)
    }
    
    sample_count = 0
    
    for split_name, (split_df, split_dir) in datasets.items():
        print(f"Creating {split_name} dataset...")
        
        for idx, row in split_df.iterrows():
            # Create synthetic multi-temporal data (6 time steps)
            temporal_data = []
            
            # Determine dominant crop for this sample
            crop_areas = {
                'RICE': row['RICE AREA (1000 ha)'],
                'WHEAT': row['WHEAT AREA (1000 ha)'],
                'MAIZE': row['MAIZE AREA (1000 ha)'],
                'SUGARCANE': row['SUGARCANE AREA (1000 ha)'],
                'COTTON': row['COTTON AREA (1000 ha)']
            }
            
            # Find dominant crop
            dominant_crop = max(crop_areas.items(), key=lambda x: x[1])[0]
            crop_label = crop_mapping.get(dominant_crop, 5)  # Default to OTHER
            
            # Skip if no significant agriculture
            if sum(crop_areas.values()) < 1.0:  # Less than 1000 ha total
                continue
            
            # Create 6 time steps (simulating monthly observations)
            for t in range(6):
                # Create synthetic optical bands (6 channels: R,G,B,NIR,SWIR1,SWIR2)
                optical_img = np.zeros((image_size[0], image_size[1], 6), dtype=np.float32)
                
                # Generate patterns based on crop type and time
                base_pattern = generate_crop_pattern(crop_label, t, image_size)
                
                # Add crop-specific spectral signatures
                for band in range(6):
                    spectral_response = get_spectral_response(crop_label, band, t)
                    noise = np.random.normal(0, 0.05, image_size)
                    optical_img[:, :, band] = base_pattern * spectral_response + noise
                
                # Clip to valid range
                optical_img = np.clip(optical_img, 0, 1)
                temporal_data.append(optical_img)
            
            # Stack temporal data: (T, H, W, C)
            temporal_stack = np.array(temporal_data)
            
            # Save as numpy array (the dataset loader will handle this)
            sample_name = f"sample_{sample_count:06d}"
            
            # Save optical data
            np.save(split_dir / f"{sample_name}_optical.npy", temporal_stack)
            
            # Create synthetic SAR data (2 channels: VV, VH)
            sar_data = np.zeros((6, image_size[0], image_size[1], 2), dtype=np.float32)
            for t in range(6):
                for pol in range(2):  # VV and VH polarizations
                    sar_response = get_sar_response(crop_label, pol, t)
                    sar_noise = np.random.normal(0, 0.1, image_size)
                    sar_data[t, :, :, pol] = base_pattern * sar_response + sar_noise
            
            sar_data = np.clip(sar_data, -1, 1)
            np.save(split_dir / f"{sample_name}_sar.npy", sar_data)
            
            # Create synthetic weather data (5 features: temp, humidity, rainfall, wind, pressure)
            weather_data = np.array([
                20 + np.random.normal(0, 5),    # temperature
                60 + np.random.normal(0, 15),   # humidity  
                5 + np.random.exponential(10),  # rainfall
                200 + np.random.normal(0, 50),  # wind
                10 + np.random.normal(0, 3)     # pressure
            ], dtype=np.float32)
            
            np.save(split_dir / f"{sample_name}_weather.npy", weather_data)
            
            # Create segmentation mask
            mask = create_segmentation_mask(crop_label, image_size)
            cv2.imwrite(str(split_dir / f"{sample_name}_mask.png"), mask.astype(np.uint8))
            
            # Save metadata
            metadata = {
                'sample_id': sample_name,
                'dominant_crop': dominant_crop,
                'crop_label': crop_label,
                'state': row['State Name'],
                'district': row['Dist Name'],
                'year': row['Year'],
                'total_area': sum(crop_areas.values()),
                'crop_areas': crop_areas
            }
            
            with open(split_dir / f"{sample_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            sample_count += 1
            
            # Limit samples for demo
            if sample_count >= 50 and split_name == 'train':
                break
            elif sample_count >= 15 and split_name in ['val', 'test']:
                break
        
        print(f"Created {len(list(split_dir.glob('*_optical.npy')))} samples for {split_name}")
    
    return sample_count

def generate_crop_pattern(crop_label, time_step, image_size):
    """Generate spatial pattern based on crop type and growth stage."""
    h, w = image_size
    
    # Create base field pattern
    x = np.linspace(0, 4*np.pi, w)
    y = np.linspace(0, 4*np.pi, h)
    X, Y = np.meshgrid(x, y)
    
    # Crop-specific patterns
    if crop_label == 0:  # Rice - rectangular fields
        pattern = 0.7 + 0.3 * np.sin(X) * np.sin(Y)
    elif crop_label == 1:  # Wheat - linear fields  
        pattern = 0.6 + 0.4 * np.sin(X/2) 
    elif crop_label == 2:  # Maize - grid pattern
        pattern = 0.5 + 0.5 * (np.sin(X) + np.sin(Y)) / 2
    elif crop_label == 3:  # Sugarcane - long strips
        pattern = 0.8 + 0.2 * np.sin(X/4)
    elif crop_label == 4:  # Cotton - scattered pattern
        pattern = 0.4 + 0.6 * np.random.random(image_size)
    else:  # Other - mixed
        pattern = 0.5 + 0.3 * np.random.random(image_size)
    
    # Add growth stage variation
    growth_factor = 0.3 + 0.7 * (time_step / 5.0)  # Growth from 30% to 100%
    pattern *= growth_factor
    
    return np.clip(pattern, 0, 1)

def get_spectral_response(crop_label, band, time_step):
    """Get crop-specific spectral response for different bands."""
    # Simplified spectral signatures
    signatures = {
        0: [0.05, 0.05, 0.08, 0.45, 0.35, 0.25],  # Rice - high NIR
        1: [0.10, 0.12, 0.15, 0.50, 0.30, 0.20],  # Wheat
        2: [0.08, 0.10, 0.12, 0.55, 0.28, 0.18],  # Maize
        3: [0.06, 0.08, 0.10, 0.60, 0.25, 0.15],  # Sugarcane
        4: [0.12, 0.15, 0.18, 0.40, 0.35, 0.30],  # Cotton
        5: [0.09, 0.11, 0.13, 0.35, 0.32, 0.28],  # Other
    }
    
    base_response = signatures.get(crop_label, signatures[5])[band]
    
    # Add temporal variation (phenology effect)
    temporal_factor = 0.8 + 0.4 * np.sin(time_step * np.pi / 3)
    
    return base_response * temporal_factor

def get_sar_response(crop_label, polarization, time_step):
    """Get SAR backscatter response."""
    # SAR responses vary by crop structure and moisture
    sar_signatures = {
        0: [-0.15, -0.25],  # Rice (flooded fields)
        1: [-0.05, -0.15],  # Wheat
        2: [-0.08, -0.18],  # Maize  
        3: [-0.02, -0.12],  # Sugarcane (tall crops)
        4: [-0.10, -0.20],  # Cotton
        5: [-0.07, -0.17],  # Other
    }
    
    base_response = sar_signatures.get(crop_label, sar_signatures[5])[polarization]
    
    # Add growth stage effect
    growth_effect = 0.1 * (time_step / 5.0)
    
    return base_response + growth_effect

def create_segmentation_mask(crop_label, image_size):
    """Create segmentation mask for the dominant crop."""
    h, w = image_size
    mask = np.full((h, w), crop_label, dtype=np.uint8)
    
    # Add some field boundaries and variations
    # Create field parcels
    num_parcels_x = np.random.randint(2, 5)
    num_parcels_y = np.random.randint(2, 5)
    
    for i in range(num_parcels_x):
        for j in range(num_parcels_y):
            x1 = int(i * w / num_parcels_x)
            x2 = int((i+1) * w / num_parcels_x)
            y1 = int(j * h / num_parcels_y)
            y2 = int((j+1) * h / num_parcels_y)
            
            # Sometimes change crop type in different parcels
            if np.random.random() < 0.2:
                parcel_crop = np.random.randint(0, 6)
                mask[y1:y2, x1:x2] = parcel_crop
    
    return mask

def analyze_dataset(df):
    """Analyze the crop dataset and create visualizations."""
    print("=== Dataset Analysis ===")
    print(f"Total records: {len(df)}")
    print(f"Years covered: {df['Year'].min()} - {df['Year'].max()}")
    print(f"States: {df['State Name'].nunique()}")
    print(f"Districts: {df['Dist Name'].nunique()}")
    
    # Analyze crop distributions
    crop_cols = [col for col in df.columns if 'AREA' in col and '(1000 ha)' in col]
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Crop area distribution by year
    plt.subplot(2, 3, 1)
    year_crops = df.groupby('Year')[crop_cols].sum()
    year_crops.plot(kind='line', ax=plt.gca())
    plt.title('Crop Areas by Year')
    plt.ylabel('Area (1000 ha)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: State-wise distribution
    plt.subplot(2, 3, 2)
    state_total = df.groupby('State Name')[crop_cols].sum().sum(axis=1)
    state_total.plot(kind='bar', ax=plt.gca())
    plt.title('Total Agricultural Area by State')
    plt.ylabel('Total Area (1000 ha)')
    plt.xticks(rotation=45)
    
    # Plot 3: Production vs Area correlation
    plt.subplot(2, 3, 3)
    total_area = df[crop_cols].sum(axis=1)
    prod_cols = [col for col in df.columns if 'PRODUCTION' in col and '(1000 tons)' in col]
    total_prod = df[prod_cols].sum(axis=1)
    plt.scatter(total_area, total_prod, alpha=0.6)
    plt.xlabel('Total Area (1000 ha)')
    plt.ylabel('Total Production (1000 tons)')
    plt.title('Production vs Area')
    
    # Plot 4: Yield trends
    plt.subplot(2, 3, 4)
    yield_cols = [col for col in df.columns if 'YIELD' in col and 'Kg per ha' in col]
    df_yields = df[['Year'] + yield_cols].groupby('Year').mean()
    df_yields.plot(kind='line', ax=plt.gca())
    plt.title('Yield Trends by Year')
    plt.ylabel('Yield (Kg per ha)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 5: Crop diversity by district
    plt.subplot(2, 3, 5)
    crop_diversity = (df[crop_cols] > 0).sum(axis=1)
    plt.hist(crop_diversity, bins=20, alpha=0.7)
    plt.xlabel('Number of Different Crops')
    plt.ylabel('Number of Districts')
    plt.title('Crop Diversity Distribution')
    
    # Plot 6: Geographic distribution
    plt.subplot(2, 3, 6)
    district_counts = df['State Name'].value_counts()
    plt.pie(district_counts.values, labels=district_counts.index, autopct='%1.1f%%')
    plt.title('Data Distribution by State')
    
    plt.tight_layout()
    plt.savefig('data/crop_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def main():
    """Main preprocessing pipeline."""
    print("Loading Crops_data.csv...")
    
    # Load the dataset
    df = pd.read_csv('data/Crops_data.csv')
    
    # Analyze dataset
    analyzed_df = analyze_dataset(df)
    
    # Create synthetic satellite data
    print("\nCreating synthetic satellite imagery...")
    sample_count = create_synthetic_satellite_images(df, 'data')
    
    print(f"\nDataset creation complete!")
    print(f"Total samples created: {sample_count}")
    print(f"Data structure:")
    print(f"  - Optical imagery: 6 time steps × 256×256×6 bands")
    print(f"  - SAR imagery: 6 time steps × 256×256×2 polarizations") 
    print(f"  - Weather data: 5 features per sample")
    print(f"  - Segmentation masks: 256×256 with 6 crop classes")
    print(f"  - Metadata: JSON files with crop information")
    
    # Create dataset summary
    summary = {
        'total_samples': sample_count,
        'image_size': [256, 256],
        'temporal_length': 6,
        'optical_bands': 6,
        'sar_channels': 2,
        'weather_features': 5,
        'num_classes': 6,
        'class_names': ['rice', 'wheat', 'maize', 'sugarcane', 'cotton', 'other'],
        'splits': {
            'train': len(list(Path('data/train').glob('*_optical.npy'))),
            'val': len(list(Path('data/val').glob('*_optical.npy'))),
            'test': len(list(Path('data/test').glob('*_optical.npy')))
        }
    }
    
    with open('data/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset summary saved to data/dataset_summary.json")

if __name__ == "__main__":
    main()

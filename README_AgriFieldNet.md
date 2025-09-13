# AMPT Crop Classification with AgriFieldNet Dataset 🌾🛰️

This project implements the **Attention-based Multi-modal Phenology and Time-series (AMPT)** model for crop classification using real satellite imagery from the **AgriFieldNet India Competition**.

## 🌟 Overview

The AMPT model combines:
- **Optical satellite data** (Sentinel-2: 6 spectral bands)
- **SAR data** (Synthetic Aperture Radar: VV/VH polarizations)  
- **Weather data** (Temperature, humidity, rainfall, wind, pressure)
- **Temporal sequences** (6-month time series)
- **TerraTorch backbone** (IBM's PrithviViT foundation model)

### Dataset: AgriFieldNet India 🇮🇳
- **Real satellite imagery** from 4 Indian states (Uttar Pradesh, Rajasthan, Odisha, Bihar)
- **13 crop classes** mapped to 6 AMPT classes
- **Sentinel-2 multispectral** imagery (12 bands, using 6 key bands)
- **Field-based labels** with precise crop boundaries

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n ampt-agri python=3.9
conda activate ampt-agri

# Install dependencies
pip install -r requirements.txt

# Install TerraTorch (if needed)
pip install git+https://github.com/IBM/terratorch.git
```

### 2. Download AgriFieldNet Dataset

```bash
# Run the data downloader (requires Radiant MLHub API key)
python scripts/download_agrifieldnet.py
```

**API Key Setup:**
1. Register at [Radiant MLHub](https://mlhub.earth/)
2. Get your API key from profile settings
3. Set environment variable: `export MLHUB_API_KEY="your_api_key"`

### 3. Train the Model

```bash
# Train AMPT on AgriFieldNet dataset
python scripts/train_agrifieldnet.py
```

### 4. View Results

Training outputs will be saved to:
- `outputs/checkpoints/` - Model weights
- `outputs/logs/` - TensorBoard logs  
- `outputs/visualizations/` - Result plots and analysis

## 📊 Dataset Structure

```
data/
├── agrifieldnet_summary.json     # Dataset metadata
├── train/                        # Training samples
│   ├── agrifield_000001_optical.npy    # (6, 256, 256, 6) - Temporal optical
│   ├── agrifield_000001_sar.npy        # (6, 256, 256, 2) - Temporal SAR
│   ├── agrifield_000001_weather.npy    # (5,) - Weather features
│   ├── agrifield_000001_mask.png       # Segmentation labels
│   └── agrifield_000001_metadata.json  # Sample metadata
├── val/                          # Validation samples
└── test/                         # Test samples (no labels)
```

## 🎯 Model Architecture

### AMPT Components:
1. **Optical Encoder** (TerraTorch PrithviViT backbone)
   - Pre-trained on massive satellite imagery
   - 303M parameters
   - Handles 6 Sentinel-2 bands

2. **SAR Encoder** 
   - Processes VV/VH polarizations
   - Captures surface roughness and moisture

3. **Weather Integration**
   - Meteorological features
   - Seasonal patterns

4. **Cross-Modal Attention**
   - Fuses optical, SAR, and weather
   - Temporal attention across 6 months

5. **Segmentation Head**
   - Pixel-level crop classification
   - 6 crop classes output

### Class Mapping:
```python
AgriFieldNet → AMPT Classes
{
    'Wheat': 'wheat',
    'Mustard': 'mustard', 
    'Maize': 'maize',
    'Gram': 'gram',
    'Sugarcane': 'sugarcane',
    'Others': 'other_crop'  # Lentil, Green pea, Garlic, etc.
}
```

## 🔧 Configuration

Key configuration parameters in `configs/config.yaml`:

```yaml
model:
  optical_channels: 6     # Sentinel-2 bands: B02,B03,B04,B08,B11,B12
  num_classes: 6          # AMPT crop classes
  image_size: 256         # Spatial resolution
  num_time_steps: 6       # Temporal sequence length

data:
  dataset: "agrifieldnet" 
  bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']  # Blue,Green,Red,NIR,SWIR1,SWIR2
  batch_size: 4           # Reduced for memory efficiency
  
training:
  epochs: 50
  learning_rate: 5e-5     # Lower LR for real data
  augmentation:           # Real satellite data augmentations
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation: 15
```

## 📈 Expected Results

### Performance Metrics:
- **Accuracy**: 75-85% (depending on crop complexity)
- **F1-Score**: 0.70-0.80 (macro-averaged)
- **IoU**: 0.60-0.75 (intersection over union)

### Challenging Classes:
- **Other crops** (diverse small crops)
- **Fallow land** (similar spectral signature)

### Strong Performance:
- **Sugarcane** (distinctive temporal pattern)
- **Wheat** (consistent growth cycle)

## 🖼️ Visualizations

The training script generates comprehensive visualizations:

### 1. Sample Visualization
```python
# RGB, False Color, SWIR composites
# SAR VV/VH polarizations  
# Weather time series
# Crop segmentation masks
```

### 2. Training Metrics
- Loss curves (train/validation)
- Accuracy progression
- F1-score improvements
- Learning rate schedule

### 3. Prediction Results
- Ground truth vs predictions
- Error maps and analysis
- Per-class performance

### 4. Class Distribution
- Dataset balance analysis
- Pixel count statistics
- Geographic distribution

## 🛰️ Sentinel-2 Bands Used

| Band | Name | Wavelength | Usage |
|------|------|------------|-------|
| B02 | Blue | 490nm | Atmospheric correction, water |
| B03 | Green | 560nm | Vegetation health |
| B04 | Red | 665nm | Chlorophyll absorption |
| B08 | NIR | 842nm | Vegetation structure, NDVI |
| B11 | SWIR1 | 1610nm | Moisture content |
| B12 | SWIR2 | 2190nm | Crop stress, soil |

## ⚡ Advanced Features

### Multi-Modal Fusion:
- **Early fusion**: Concatenate modalities
- **Late fusion**: Separate processing + attention
- **Cross-modal attention**: Dynamic feature weighting

### Temporal Modeling:
- **LSTM**: Sequential dependencies
- **Transformer**: Long-range temporal attention
- **ConvLSTM**: Spatial-temporal convolutions

### Data Augmentation:
- **Spectral**: Band mixing, noise injection
- **Spatial**: Rotation, flip, crop
- **Temporal**: Time series perturbation

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.yaml
   data.batch_size: 2
   training.accumulate_grad_batches: 8
   ```

2. **Dataset Download Fails**
   ```bash
   # Check API key
   echo $MLHUB_API_KEY
   
   # Manual download from Radiant MLHub website
   # Place files in data/train/, data/val/, data/test/
   ```

3. **Import Errors**
   ```bash
   # Ensure TerraTorch is installed
   pip install terratorch
   
   # Check GDAL/Rasterio installation
   conda install -c conda-forge gdal rasterio
   ```

4. **Low Performance**
   - Check class balance (use weighted loss)
   - Increase temporal sequence length
   - Add more data augmentation
   - Fine-tune learning rate

## 📚 References

1. **AgriFieldNet Competition**: [Radiant Earth Foundation](https://mlhub.earth/data/ref_agrifieldnet_competition_v1)
2. **TerraTorch**: [IBM's Geospatial Foundation Models](https://github.com/IBM/terratorch)
3. **PrithviViT**: [Foundation Model for Earth Observation](https://arxiv.org/abs/2301.04944)
4. **Sentinel-2**: [ESA Copernicus Mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Radiant Earth Foundation** for the AgriFieldNet dataset
- **IBM** for the TerraTorch foundation models
- **ESA** for Sentinel-2 satellite imagery
- **Indian agricultural research** community

---

**Happy Crop Classification! 🌾🚀**

For questions or support, please open an issue on GitHub.

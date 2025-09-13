# AMPT (Adaptive Multi-Modal Phenological Transformer) for Crop Classification

## Overview
AMPT is a deep learning framework designed for crop classification using multi-modal satellite data. It leverages PyTorch Lightning and TerraTorch to implement a novel cross-modal phenological attention mechanism that dynamically weights SAR and optical satellite data based on crop growth stages. The project is built to handle the AgriFieldNet competition dataset, which includes six crop classes: gram, maize, mustard, sugarcane, wheat, and other crops.

## Project Structure
```
ampt-crop-classification/
├── src/                     # Source code for the project
│   ├── models/              # Model definitions
│   ├── data/                # Data loading and processing
│   ├── losses/              # Loss functions
│   ├── training/            # Training logic and callbacks
│   └── utils/               # Utility functions
├── configs/                 # Configuration files
├── scripts/                 # Scripts for training, evaluation, and submission
├── tests/                   # Unit and integration tests
├── data/                    # Dataset directories (train, val, test)
├── outputs/                 # Outputs (checkpoints, logs, submissions)
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation
└── README.md                # Project documentation
```

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
1. **Training the Model**: Use the `train_model.py` script to start training. Configuration options can be set in `configs/config.yaml`.
   ```
   python scripts/train_model.py
   ```

2. **Evaluating the Model**: After training, evaluate the model on the test set using:
   ```
   python scripts/evaluate_model.py
   ```

3. **Creating Submission**: Generate the submission files required for the competition with:
   ```
   python scripts/create_submission.py
   ```

## Features
- Multi-temporal satellite data processing (6 months)
- Fusion of SAR, optical, and weather data
- Selective loss computation for sparse labels
- 6-class crop segmentation
- Comprehensive logging and visualization

## Acknowledgments
This project utilizes the AgriFieldNet competition dataset and is built upon the foundations of PyTorch Lightning and TerraTorch. Special thanks to the contributors of these libraries for their invaluable tools and resources.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
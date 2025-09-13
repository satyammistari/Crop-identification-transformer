"""
Train AMPT Model on Real AgriFieldNet Dataset
Enhanced training script with proper data handling and visualization capabilities.
"""

import os
import sys
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ampt_model import AMPTModel
from src.data.agrifieldnet_dataset import AgriFieldNetDataset, create_agrifieldnet_dataloaders
from scripts.download_agrifieldnet import AgriFieldNetDownloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMPTDataModule(pl.LightningDataModule):
    """Lightning Data Module for AgriFieldNet dataset."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.data_dir = config.data.data_dir
        
        # Dataset parameters
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.image_size = config.model.image_size
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self, stage: str = None):
        """Setup data loaders."""
        logger.info(f"Setting up data module for stage: {stage}")
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_agrifieldnet_dataloaders(
            self.data_dir, 
            {
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'image_size': self.image_size,
                'use_preprocessed': True
            }
        )
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader

def visualize_training_results(trainer, model, data_module, output_dir):
    """Create comprehensive training visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating training result visualizations...")
    
    # Get a validation batch for visualization
    model.eval()
    val_loader = data_module.val_dataloader()
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Create prediction visualizations
            create_prediction_visualizations(
                batch, outputs, output_dir, num_samples=4
            )
            break
    
    # Plot training metrics if available
    if hasattr(trainer.logger, 'log_dir'):
        plot_training_metrics(trainer.logger.log_dir, output_dir)
    
    logger.info(f"Visualizations saved to {output_dir}")

def create_prediction_visualizations(batch, outputs, output_dir, num_samples=4):
    """Create prediction result visualizations."""
    
    batch_size = min(num_samples, batch['optical'].size(0))
    
    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Get optical image (take middle time step and first 3 bands for RGB)
        optical = batch['optical'][i, 0, :3].permute(1, 2, 0).cpu().numpy()
        optical = np.clip(optical * 0.5 + 0.5, 0, 1)  # Denormalize
        
        # Get true mask
        if 'mask' in batch:
            true_mask = batch['mask'][i].cpu().numpy()
        else:
            true_mask = np.zeros((256, 256))
        
        # Get predicted mask
        if 'segmentation_logits' in outputs:
            pred_logits = outputs['segmentation_logits'][i]
            pred_mask = torch.argmax(pred_logits, dim=0).cpu().numpy()
        else:
            pred_mask = np.zeros_like(true_mask)
        
        # Plot RGB image
        axes[i, 0].imshow(optical)
        axes[i, 0].set_title(f'Sample {i+1}: RGB Image')
        axes[i, 0].axis('off')
        
        # Plot true mask
        im1 = axes[i, 1].imshow(true_mask, cmap='tab10', vmin=0, vmax=5)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot predicted mask
        im2 = axes[i, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=5)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Plot difference
        diff = (pred_mask != true_mask).astype(float)
        im3 = axes[i, 3].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[i, 3].set_title('Prediction Error')
        axes[i, 3].axis('off')
    
    # Add colorbar
    class_names = ['Gram', 'Maize', 'Mustard', 'Sugarcane', 'Wheat', 'Other']
    cbar = plt.colorbar(im1, ax=axes[:, 1], fraction=0.046, pad=0.04)
    cbar.set_ticks(range(6))
    cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_metrics(log_dir, output_dir):
    """Plot training metrics from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find the latest event file
        event_files = list(Path(log_dir).rglob('events.out.tfevents.*'))
        if not event_files:
            logger.warning("No TensorBoard event files found")
            return
        
        latest_event_file = max(event_files, key=lambda x: x.stat().st_mtime)
        
        # Load events
        ea = EventAccumulator(str(latest_event_file))
        ea.Reload()
        
        # Get available scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('train_loss', 'Training Loss'),
            ('val_loss', 'Validation Loss'),
            ('val_accuracy', 'Validation Accuracy'),
            ('val_f1_score', 'Validation F1 Score')
        ]
        
        for idx, (metric_name, title) in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
                
            if metric_name in scalar_tags:
                steps, values = zip(*[(s.step, s.value) for s in ea.Scalars(metric_name)])
                axes[idx].plot(steps, values)
                axes[idx].set_title(title)
                axes[idx].set_xlabel('Step')
                axes[idx].set_ylabel('Value')
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, f'{metric_name}\\nNot Available', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(title)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        logger.warning("TensorBoard not available for metrics plotting")
    except Exception as e:
        logger.warning(f"Could not plot training metrics: {e}")

def create_class_distribution_plot(data_module, output_dir):
    """Create class distribution visualization."""
    logger.info("Analyzing class distribution...")
    
    # Get dataset
    train_dataset = AgriFieldNetDataset(
        data_module.data_dir, 
        'train',
        config={'use_preprocessed': True}
    )
    
    if len(train_dataset) == 0:
        logger.warning("No training samples found for class distribution analysis")
        return
    
    # Count classes
    class_counts = np.zeros(6)
    total_pixels = 0
    
    for i in range(min(len(train_dataset), 50)):  # Sample subset for speed
        try:
            sample = train_dataset[i]
            if 'mask' in sample:
                mask = sample['mask'].numpy()
                for c in range(6):
                    class_counts[c] += (mask == c).sum()
                total_pixels += mask.size
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    class_names = ['Gram', 'Maize', 'Mustard', 'Sugarcane', 'Wheat', 'Other']
    
    # Bar plot of class counts
    ax1.bar(class_names, class_counts)
    ax1.set_title('Class Distribution (Pixel Counts)')
    ax1.set_ylabel('Number of Pixels')
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart of class percentages
    percentages = class_counts / class_counts.sum() * 100
    ax2.pie(percentages, labels=class_names, autopct='%1.1f%%')
    ax2.set_title('Class Distribution (Percentages)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Class distribution: {dict(zip(class_names, class_counts))}")

def setup_data_if_needed(config):
    """Download and setup AgriFieldNet data if not available."""
    data_dir = Path(config.data.data_dir)
    
    # Check if preprocessed data exists
    train_dir = data_dir / 'train'
    if train_dir.exists() and len(list(train_dir.glob('*_optical.npy'))) > 0:
        logger.info("Preprocessed data found")
        return True
    
    # Check if raw AgriFieldNet data exists
    if train_dir.exists() and len(list(train_dir.glob('*.tif'))) > 0:
        logger.info("Raw AgriFieldNet data found")
        return True
    
    # Try to download data
    logger.info("No data found. Attempting to download AgriFieldNet dataset...")
    
    try:
        downloader = AgriFieldNetDownloader(data_dir)
        summary = downloader.process_full_dataset()
        logger.info(f"Dataset download completed: {summary['splits']}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("Please manually download the dataset or run:")
        logger.info("python scripts/download_agrifieldnet.py")
        return False

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """Main training function."""
    logger.info("Starting AMPT training on AgriFieldNet dataset")
    logger.info(f"Configuration: {config}")
    
    # Setup output directories
    output_dir = Path(config.output.checkpoint_dir).parent
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data
    if not setup_data_if_needed(config):
        logger.error("Data setup failed. Cannot proceed with training.")
        return
    
    # Initialize data module
    data_module = AMPTDataModule(config)
    data_module.setup()
    
    # Create class distribution plot
    create_class_distribution_plot(data_module, viz_dir)
    
    # Check if we have training data
    if data_module.train_loader is None or len(data_module.train_loader) == 0:
        logger.error("No training data available. Please check data setup.")
        return
    
    logger.info(f"Training data: {len(data_module.train_loader)} batches")
    logger.info(f"Validation data: {len(data_module.val_loader)} batches")
    
    # Initialize model
    model = AMPTModel(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.output.checkpoint_dir,
            filename='ampt-agrifieldnet-{epoch:02d}-{val_f1_score:.3f}',
            monitor='val_f1_score',
            mode='max',
            save_top_k=5,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_f1_score',
            patience=15,
            mode='max',
            min_delta=0.001
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=config.logging.log_dir,
        name='ampt_agrifieldnet',
        version=None
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        val_check_interval=config.logging.val_check_interval,
        gradient_clip_val=config.training.gradient_clip,
        accumulate_grad_batches=config.training.get('accumulate_grad_batches', 1)
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test best model
    logger.info("Testing best model...")
    trainer.test(ckpt_path='best', datamodule=data_module)
    
    # Create visualizations
    visualize_training_results(trainer, model, data_module, viz_dir)
    
    # Save model summary
    model_summary = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'config': dict(config),
        'best_checkpoint': trainer.checkpoint_callback.best_model_path,
        'final_metrics': trainer.logged_metrics
    }
    
    import json
    with open(viz_dir / 'model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2, default=str)
    
    logger.info("Training completed!")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"Final metrics: {trainer.logged_metrics}")
    logger.info(f"Visualizations saved to: {viz_dir}")
    
    print("\\n" + "="*60)
    print("🎉 AMPT TRAINING ON AGRIFIELDNET COMPLETED! 🎉")
    print("="*60)
    print(f"📊 Total Parameters: {model_summary['total_parameters']:,}")
    print(f"🏆 Best Checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"📈 Visualizations: {viz_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

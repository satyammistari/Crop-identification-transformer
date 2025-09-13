#!/usr/bin/env python3
"""
Training script for AMPT model.

This script handles the complete training pipeline including:
- Configuration management with Hydra
- Model initialization and setup
- Data loading and augmentation
- Training loop with callbacks
- Logging and checkpointing
- Distributed training support

Example usage:
    python scripts/train_model.py
    python scripts/train_model.py model.backbone=resnet50 trainer.max_epochs=100
    python scripts/train_model.py --config-name=production_config
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
    DeviceStatsMonitor, ProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ampt_model import AMPTModel
from src.data.agrifieldnet_datamodule import AgriFieldNetDataModule
# from src.training.callbacks import (
#     AttentionVisualizationCallback,
#     PredictionVisualizationCallback,
#     ModelComplexityCallback
# )
from src.utils.metrics import SegmentationMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logger(config: DictConfig) -> Optional[pl.loggers.Logger]:
    """
    Setup experiment logger (WandB or TensorBoard).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger or None
    """
    logger_config = config.get('logger', {})
    logger_type = logger_config.get('type', 'tensorboard')
    
    if logger_type == 'wandb':
        return WandbLogger(
            project=logger_config.get('project', 'ampt-crop-classification'),
            name=logger_config.get('experiment_name', 'default'),
            save_dir=logger_config.get('save_dir', 'outputs/logs'),
            log_model=logger_config.get('log_model', True),
            config=OmegaConf.to_container(config, resolve=True)
        )
    elif logger_type == 'tensorboard':
        return TensorBoardLogger(
            save_dir=logger_config.get('save_dir', 'outputs/logs'),
            name=logger_config.get('experiment_name', 'default'),
            log_graph=logger_config.get('log_graph', True)
        )
    else:
        logger.warning(f"Unknown logger type: {logger_type}. No logger will be used.")
        return None


def setup_callbacks(config: DictConfig) -> List[pl.Callback]:
    """
    Setup training callbacks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_config = config.get('checkpoint', {})
    if checkpoint_config.get('enabled', True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_config.get('dirpath', 'outputs/checkpoints'),
            filename=checkpoint_config.get('filename', 'ampt-{epoch:02d}-{val_miou:.3f}'),
            monitor=checkpoint_config.get('monitor', 'val_miou'),
            mode=checkpoint_config.get('mode', 'max'),
            save_top_k=checkpoint_config.get('save_top_k', 3),
            save_last=checkpoint_config.get('save_last', True),
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_config = config.get('early_stopping', {})
    if early_stop_config.get('enabled', True):
        early_stopping = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_miou'),
            mode=early_stop_config.get('mode', 'max'),
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitoring
    if config.get('monitor_lr', True):
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    # Device stats monitoring
    if config.get('monitor_device', True):
        callbacks.append(DeviceStatsMonitor())
    
    # Custom callbacks
    custom_callbacks_config = config.get('custom_callbacks', {})
    
    # Attention visualization
    # if custom_callbacks_config.get('attention_viz', False):
    #     callbacks.append(AttentionVisualizationCallback(
    #         log_every_n_epochs=custom_callbacks_config.get('attention_viz_frequency', 5)
    #     ))
    
    # Prediction visualization
    # if custom_callbacks_config.get('prediction_viz', False):
    #     callbacks.append(PredictionVisualizationCallback(
    #         log_every_n_epochs=custom_callbacks_config.get('prediction_viz_frequency', 10),
    #         num_samples=custom_callbacks_config.get('prediction_viz_samples', 4)
    #     ))
    
    # Model complexity monitoring
    # if custom_callbacks_config.get('model_complexity', False):
    #     callbacks.append(ModelComplexityCallback())
    
    return callbacks


def setup_trainer(config: DictConfig, logger_instance: Optional[pl.loggers.Logger], 
                 callbacks: List[pl.Callback]) -> pl.Trainer:
    """
    Setup PyTorch Lightning trainer.
    
    Args:
        config: Configuration dictionary
        logger_instance: Configured logger
        callbacks: List of callbacks
        
    Returns:
        Configured trainer
    """
    trainer_config = config.get('trainer', {})
    
    # GPU/device configuration
    accelerator = trainer_config.get('accelerator', 'auto')
    devices = trainer_config.get('devices', 'auto')
    
    # Precision and performance
    precision = trainer_config.get('precision', '16-mixed')
    accumulate_grad_batches = trainer_config.get('accumulate_grad_batches', 1)
    
    # Training configuration
    max_epochs = trainer_config.get('max_epochs', 100)
    max_steps = trainer_config.get('max_steps', -1)
    
    # Validation and logging
    val_check_interval = trainer_config.get('val_check_interval', 1.0)
    log_every_n_steps = trainer_config.get('log_every_n_steps', 50)
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        logger=logger_instance,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=trainer_config.get('deterministic', False),
        benchmark=trainer_config.get('benchmark', True),
        fast_dev_run=trainer_config.get('fast_dev_run', False),
        limit_train_batches=trainer_config.get('limit_train_batches', 1.0),
        limit_val_batches=trainer_config.get('limit_val_batches', 1.0),
        limit_test_batches=trainer_config.get('limit_test_batches', 1.0),
        num_sanity_val_steps=trainer_config.get('num_sanity_val_steps', 2),
        profiler=trainer_config.get('profiler', None)
    )
    
    return trainer


def validate_config(config: DictConfig) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['model', 'data', 'trainer']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data paths
    data_config = config.data
    data_dir = Path(data_config.get('data_dir', 'data'))
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Check for train/val directories
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    if not train_dir.exists():
        raise ValueError(f"Training data directory does not exist: {train_dir}")
    if not val_dir.exists():
        raise ValueError(f"Validation data directory does not exist: {val_dir}")
    
    # Validate model parameters
    model_config = config.model
    if model_config.get('num_classes', 6) <= 0:
        raise ValueError("Number of classes must be positive")
    
    if model_config.get('optical_channels', 13) <= 0:
        raise ValueError("Number of optical channels must be positive")
    
    logger.info("Configuration validation passed")


def setup_environment(config: DictConfig) -> None:
    """
    Setup training environment.
    
    Args:
        config: Configuration dictionary
    """
    # Set random seeds
    if 'seed' in config:
        pl.seed_everything(config.seed, workers=True)
        logger.info(f"Set random seed to {config.seed}")
    
    # Create output directories
    output_dirs = [
        'outputs/checkpoints',
        'outputs/logs',
        'outputs/submissions',
        'outputs/visualizations'
    ]
    
    for dir_path in output_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Directory already exists, continue
            pass
    
    # Set environment variables for performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Configure torch
    torch.set_float32_matmul_precision('medium')
    
    logger.info("Environment setup completed")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Hydra configuration
    """
    try:
        # Setup
        logger.info("Starting AMPT training...")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
        
        # Validate configuration and setup environment
        validate_config(config)
        setup_environment(config)
        
        # Initialize data module
        logger.info("Initializing data module...")
        datamodule = AgriFieldNetDataModule(config.data)
        datamodule.setup('fit')
        
        # Log data statistics
        logger.info(f"Training samples: {len(datamodule.train_dataset)}")
        logger.info(f"Validation samples: {len(datamodule.val_dataset)}")
        logger.info(f"Batch size: {config.data.batch_size}")
        logger.info(f"Number of workers: {config.data.num_workers}")
        
        # Initialize model
        logger.info("Initializing model...")
        model = AMPTModel(config.model)
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Setup training components
        logger_instance = setup_logger(config)
        callbacks = setup_callbacks(config)
        trainer = setup_trainer(config, logger_instance, callbacks)
        
        # Resume from checkpoint if specified
        ckpt_path = config.get('resume_from_checkpoint', None)
        if ckpt_path and Path(ckpt_path).exists():
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            ckpt_path = None
        
        # Start training
        logger.info("Starting training...")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        
        # Optionally run testing
        if config.get('run_test_after_training', False):
            logger.info("Running testing...")
            datamodule.setup('test')
            trainer.test(model, datamodule=datamodule)
        
        # Save final model
        if config.get('save_final_model', True):
            final_model_path = Path('outputs/checkpoints/final_model.ckpt')
            trainer.save_checkpoint(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
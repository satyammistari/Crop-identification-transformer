"""
Training system for AMPT crop classification model.

This module provides comprehensive training infrastructure including:
1. Lightning trainer configuration
2. Custom callbacks for monitoring and checkpointing
3. Learning rate scheduling
4. Early stopping and model selection
5. Training orchestration and experiment tracking
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    DeviceStatsMonitor, RichProgressBar, RichModelSummary
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)
import numpy as np
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
import logging
from datetime import datetime
import json
import warnings
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AMPTTrainingCallback(pl.Callback):
    """
    Custom callback for AMPT model training.
    
    Handles specialized logging, attention visualization,
    and model-specific monitoring during training.
    """
    
    def __init__(
        self,
        log_attention: bool = True,
        attention_frequency: int = 10,
        save_attention_maps: bool = False,
        visualize_predictions: bool = True,
        prediction_frequency: int = 50
    ):
        super().__init__()
        self.log_attention = log_attention
        self.attention_frequency = attention_frequency
        self.save_attention_maps = save_attention_maps
        self.visualize_predictions = visualize_predictions
        self.prediction_frequency = prediction_frequency
        
        self.step_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training metrics and attention maps."""
        self.step_count += 1
        
        # Log attention maps periodically
        if (self.log_attention and 
            self.step_count % self.attention_frequency == 0 and 
            hasattr(pl_module, 'last_attention_weights')):
            
            self._log_attention_weights(trainer, pl_module)
        
        # Log custom metrics
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.startswith('loss_'):
                    trainer.logger.log_metrics({f"train/{key}": value}, step=trainer.global_step)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Log validation metrics and visualizations."""
        if (self.visualize_predictions and 
            batch_idx % self.prediction_frequency == 0 and 
            batch_idx < 5):  # Only first few batches
            
            self._log_prediction_visualization(trainer, pl_module, batch, outputs)
    
    def _log_attention_weights(self, trainer, pl_module):
        """Log attention weight visualizations."""
        try:
            if hasattr(pl_module.model, 'cross_modal_attention'):
                attention_weights = pl_module.model.cross_modal_attention.get_attention_weights()
                
                if attention_weights is not None and trainer.logger:
                    # Create attention heatmap
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    im = ax.imshow(attention_weights.cpu().numpy(), cmap='viridis')
                    ax.set_title('Cross-Modal Attention Weights')
                    ax.set_xlabel('Spatial Dimension')
                    ax.set_ylabel('Temporal Dimension')
                    plt.colorbar(im)
                    
                    if isinstance(trainer.logger, WandbLogger):
                        trainer.logger.log_image('attention/cross_modal', [fig])
                    
                    plt.close(fig)
                    
        except Exception as e:
            logger.warning(f"Failed to log attention weights: {e}")
    
    def _log_prediction_visualization(self, trainer, pl_module, batch, outputs):
        """Log prediction visualizations."""
        try:
            if not isinstance(trainer.logger, WandbLogger):
                return
            
            # Get first sample from batch
            optical = batch['optical'][0]  # [T, C, H, W]
            if 'mask' in batch:
                mask = batch['mask'][0]  # [H, W]
            else:
                mask = None
            
            if isinstance(outputs, dict) and 'segmentation' in outputs:
                pred_logits = outputs['segmentation'][0]  # [C, H, W]
                pred = torch.argmax(pred_logits, dim=0)
            else:
                return
            
            # Create visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image (last timestep)
            img = optical[-1, :3].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            axes[0].imshow(img)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Prediction
            pred_vis = pred.cpu().numpy()
            axes[1].imshow(pred_vis, cmap='tab10', vmin=0, vmax=5)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            
            # Ground truth (if available)
            if mask is not None:
                axes[2].imshow(mask.cpu().numpy(), cmap='tab10', vmin=0, vmax=5)
                axes[2].set_title('Ground Truth')
            else:
                axes[2].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].axis('off')
            
            trainer.logger.log_image('predictions/sample', [fig])
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to log prediction visualization: {e}")


class AMPTModelCheckpoint(ModelCheckpoint):
    """
    Enhanced model checkpoint callback for AMPT.
    
    Adds support for saving best models based on multiple metrics
    and model architecture-specific checkpoint handling.
    """
    
    def __init__(
        self,
        dirpath: str,
        filename: str = 'ampt-{epoch:02d}-{val_loss:.3f}',
        monitor: str = 'val_loss',
        save_top_k: int = 3,
        save_last: bool = True,
        save_weights_only: bool = False,
        auto_insert_metric_name: bool = False,
        **kwargs
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            save_top_k=save_top_k,
            save_last=save_last,
            save_weights_only=save_weights_only,
            auto_insert_metric_name=auto_insert_metric_name,
            **kwargs
        )
        
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Add custom information to checkpoint."""
        # Add model configuration
        if hasattr(pl_module, 'config'):
            checkpoint['model_config'] = OmegaConf.to_container(pl_module.config)
        
        # Add training metadata
        checkpoint['training_metadata'] = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add attention weights if available
        if hasattr(pl_module.model, 'cross_modal_attention'):
            try:
                attention_state = pl_module.model.cross_modal_attention.state_dict()
                checkpoint['attention_weights'] = attention_state
            except Exception as e:
                logger.warning(f"Failed to save attention weights: {e}")
        
        return checkpoint


class AMPTLearningRateScheduler:
    """
    Learning rate scheduler factory for AMPT training.
    
    Provides various scheduling strategies optimized for crop classification.
    """
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'cosine',
        max_epochs: int = 100,
        steps_per_epoch: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create learning rate scheduler configuration.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('cosine', 'plateau', 'onecycle', 'warmrestarts')
            max_epochs: Maximum training epochs
            steps_per_epoch: Steps per epoch for step-based schedulers
            **kwargs: Additional scheduler parameters
        
        Returns:
            Dict[str, Any]: Scheduler configuration for Lightning
        """
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=kwargs.get('eta_min', 1e-7)
            )
            return {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                threshold=kwargs.get('threshold', 1e-4),
                min_lr=kwargs.get('min_lr', 1e-7)
            )
            return {
                'scheduler': scheduler,
                'monitor': kwargs.get('monitor', 'val_loss'),
                'interval': 'epoch',
                'frequency': 1
            }
        
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', 1e-3),
                total_steps=max_epochs * steps_per_epoch,
                pct_start=kwargs.get('pct_start', 0.3),
                div_factor=kwargs.get('div_factor', 25),
                final_div_factor=kwargs.get('final_div_factor', 10000)
            )
            return {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        
        elif scheduler_type == 'warmrestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 1e-7)
            )
            return {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class AMPTTrainer:
    """
    High-level trainer for AMPT crop classification model.
    
    Provides a complete training pipeline with optimal defaults
    for agricultural satellite imagery classification.
    """
    
    def __init__(
        self,
        config: DictConfig,
        experiment_name: str = None,
        project_name: str = "ampt-crop-classification",
        tags: List[str] = None
    ):
        self.config = config
        self.experiment_name = experiment_name or f"ampt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.project_name = project_name
        self.tags = tags or []
        
        # Setup directories
        self.output_dir = Path(config.trainer.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.experiment_name
        self.log_dir = self.output_dir / "logs" / self.experiment_name
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AMPTTrainer initialized for experiment: {self.experiment_name}")
    
    def setup_callbacks(self) -> List[pl.Callback]:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = AMPTModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename='best-{epoch:02d}-{val_iou:.3f}',
            monitor='val_iou',
            mode='max',
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.trainer.get('early_stopping', {}).get('enabled', True):
            early_stop_callback = EarlyStopping(
                monitor=self.config.trainer.early_stopping.get('monitor', 'val_loss'),
                patience=self.config.trainer.early_stopping.get('patience', 20),
                mode=self.config.trainer.early_stopping.get('mode', 'min'),
                min_delta=self.config.trainer.early_stopping.get('min_delta', 1e-4),
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # Custom AMPT callback
        ampt_callback = AMPTTrainingCallback(
            log_attention=self.config.trainer.get('log_attention', True),
            attention_frequency=self.config.trainer.get('attention_frequency', 50),
            visualize_predictions=self.config.trainer.get('visualize_predictions', True),
            prediction_frequency=self.config.trainer.get('prediction_frequency', 100)
        )
        callbacks.append(ampt_callback)
        
        # Device stats monitor
        if self.config.trainer.get('monitor_hardware', False):
            device_stats = DeviceStatsMonitor()
            callbacks.append(device_stats)
        
        # Rich progress bar
        if self.config.trainer.get('use_rich_progress', True):
            progress_bar = RichProgressBar()
            callbacks.append(progress_bar)
        
        # Model summary
        model_summary = RichModelSummary(max_depth=3)
        callbacks.append(model_summary)
        
        return callbacks
    
    def setup_logger(self) -> Union[WandbLogger, TensorBoardLogger]:
        """Setup experiment logger."""
        logger_type = self.config.trainer.get('logger', 'wandb')
        
        if logger_type == 'wandb':
            return WandbLogger(
                project=self.project_name,
                name=self.experiment_name,
                tags=self.tags,
                save_dir=str(self.log_dir),
                log_model=True,
                config=OmegaConf.to_container(self.config, resolve=True)
            )
        
        elif logger_type == 'tensorboard':
            return TensorBoardLogger(
                save_dir=str(self.log_dir),
                name=self.experiment_name
            )
        
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")
    
    def setup_profiler(self):
        """Setup performance profiler."""
        profiler_type = self.config.trainer.get('profiler', None)
        
        if profiler_type == 'simple':
            return SimpleProfiler(
                dirpath=str(self.log_dir),
                filename='profile'
            )
        elif profiler_type == 'advanced':
            return AdvancedProfiler(
                dirpath=str(self.log_dir),
                filename='profile'
            )
        else:
            return None
    
    def create_trainer(self) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        # Setup components
        callbacks = self.setup_callbacks()
        logger = self.setup_logger()
        profiler = self.setup_profiler()
        
        # Trainer configuration
        trainer_config = {
            'max_epochs': self.config.trainer.max_epochs,
            'min_epochs': self.config.trainer.get('min_epochs', 1),
            'callbacks': callbacks,
            'logger': logger,
            'profiler': profiler,
            'enable_checkpointing': True,
            'log_every_n_steps': self.config.trainer.get('log_every_n_steps', 50),
            'val_check_interval': self.config.trainer.get('val_check_interval', 1.0),
            'limit_train_batches': self.config.trainer.get('limit_train_batches', 1.0),
            'limit_val_batches': self.config.trainer.get('limit_val_batches', 1.0),
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'deterministic': self.config.trainer.get('deterministic', False),
            'benchmark': self.config.trainer.get('benchmark', True)
        }
        
        # GPU configuration
        if torch.cuda.is_available():
            trainer_config.update({
                'accelerator': 'gpu',
                'devices': self.config.trainer.get('devices', 1),
                'strategy': self.config.trainer.get('strategy', 'auto'),
                'precision': self.config.trainer.get('precision', '16-mixed')
            })
        else:
            trainer_config.update({
                'accelerator': 'cpu',
                'precision': '32'
            })
        
        # Gradient clipping
        if self.config.trainer.get('gradient_clip_val', 0) > 0:
            trainer_config['gradient_clip_val'] = self.config.trainer.gradient_clip_val
            trainer_config['gradient_clip_algorithm'] = self.config.trainer.get('gradient_clip_algorithm', 'norm')
        
        # Accumulate gradients
        if self.config.trainer.get('accumulate_grad_batches', 1) > 1:
            trainer_config['accumulate_grad_batches'] = self.config.trainer.accumulate_grad_batches
        
        return pl.Trainer(**trainer_config)
    
    def train(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the AMPT model.
        
        Args:
            model: Lightning module to train
            datamodule: Lightning data module
            resume_from_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Starting training for experiment: {self.experiment_name}")
        
        # Save config
        config_file = self.log_dir / "config.yaml"
        OmegaConf.save(self.config, config_file)
        
        # Create trainer
        trainer = self.create_trainer()
        
        # Start training
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=resume_from_checkpoint
        )
        
        # Get training results
        results = {
            'best_model_path': trainer.checkpoint_callback.best_model_path,
            'best_model_score': trainer.checkpoint_callback.best_model_score.item(),
            'experiment_name': self.experiment_name,
            'final_epoch': trainer.current_epoch,
            'global_step': trainer.global_step
        }
        
        logger.info(f"Training completed. Best model: {results['best_model_path']}")
        
        return results
    
    def test(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        checkpoint_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Test the AMPT model.
        
        Args:
            model: Lightning module to test
            datamodule: Lightning data module
            checkpoint_path: Path to checkpoint to load
        
        Returns:
            List[Dict[str, Any]]: Test results
        """
        logger.info("Starting model testing")
        
        # Create trainer for testing
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True
        )
        
        # Run test
        test_results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path
        )
        
        logger.info("Testing completed")
        
        return test_results


# Export for easy imports
__all__ = ['AMPTTrainer', 'AMPTTrainingCallback', 'AMPTModelCheckpoint', 'AMPTLearningRateScheduler']

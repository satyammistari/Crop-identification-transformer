"""
SegmentationMetrics: Comprehensive evaluation metrics for crop classification.

This module provides comprehensive metrics for evaluating segmentation
performance including:
- IoU (Intersection over Union) per class and mean
- Accuracy (pixel-level and class-averaged)
- F1-score per class and macro-averaged
- Precision and Recall per class
- Confusion matrix computation
- Support for sparse labeling (ignore pixels)
"""

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')


class SegmentationMetrics(Metric):
    """
    Comprehensive segmentation metrics with support for sparse labeling.
    
    This metric class computes multiple evaluation metrics for segmentation
    tasks while properly handling ignore pixels and class imbalance.
    
    Args:
        num_classes (int): Number of classes
        class_names (List[str]): Names of classes for reporting
        ignore_index (int): Index to ignore in metric computation
        compute_confusion_matrix (bool): Whether to compute confusion matrix
        average (str): Averaging method for multi-class metrics
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        compute_confusion_matrix: bool = True,
        average: str = 'macro'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.compute_confusion_matrix = compute_confusion_matrix
        self.average = average
        
        # Set default class names if not provided
        if class_names is None:
            self.class_names = [f'class_{i}' for i in range(num_classes)]
        else:
            self.class_names = class_names
        
        # Initialize metric states
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("support", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        
        if self.compute_confusion_matrix:
            self.add_state(
                "confusion_matrix", 
                default=torch.zeros(num_classes, num_classes), 
                dist_reduce_fx="sum"
            )
        
        # Total pixels for overall accuracy
        self.add_state("total_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metric states with new predictions and targets.
        
        Args:
            preds (torch.Tensor): Predictions [B, C, H, W] or [B, H, W]
            target (torch.Tensor): Ground truth labels [B, H, W]
        """
        # Handle different input formats
        if preds.dim() == 4:  # [B, C, H, W] - logits
            preds = torch.argmax(preds, dim=1)
        elif preds.dim() == 3:  # [B, H, W] - already class indices
            pass
        else:
            raise ValueError(f"Predictions must be 3D or 4D, got {preds.dim()}D")
        
        # Flatten tensors
        preds_flat = preds.flatten()
        target_flat = target.flatten()
        
        # Create valid mask (ignore pixels with ignore_index)
        valid_mask = target_flat != self.ignore_index
        
        if not valid_mask.any():
            # No valid pixels, skip this batch
            return
        
        # Apply valid mask
        preds_valid = preds_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Ensure predictions are in valid range
        preds_valid = torch.clamp(preds_valid, 0, self.num_classes - 1)
        target_valid = torch.clamp(target_valid, 0, self.num_classes - 1)
        
        # Update overall accuracy
        correct = (preds_valid == target_valid).sum()
        self.total_correct += correct
        self.total_pixels += len(preds_valid)
        
        # Update per-class metrics
        for class_id in range(self.num_classes):
            pred_class = (preds_valid == class_id)
            target_class = (target_valid == class_id)
            
            tp = (pred_class & target_class).sum()
            fp = (pred_class & ~target_class).sum()
            fn = (~pred_class & target_class).sum()
            support = target_class.sum()
            
            self.true_positives[class_id] += tp
            self.false_positives[class_id] += fp
            self.false_negatives[class_id] += fn
            self.support[class_id] += support
        
        # Update confusion matrix
        if self.compute_confusion_matrix:
            # Use sklearn's confusion_matrix for efficiency
            cm = confusion_matrix(
                target_valid.cpu().numpy(),
                preds_valid.cpu().numpy(),
                labels=list(range(self.num_classes))
            )
            self.confusion_matrix += torch.tensor(cm, device=self.confusion_matrix.device)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final metric values.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of computed metrics
        """
        metrics = {}
        
        # Overall accuracy
        overall_accuracy = self.total_correct.float() / (self.total_pixels + 1e-8)
        metrics['accuracy'] = overall_accuracy
        
        # Per-class metrics
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = self.true_positives / (
            self.true_positives + self.false_positives + self.false_negatives + 1e-8
        )
        
        # Store per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1_score[i]
            metrics[f'iou_{class_name}'] = iou[i]
        
        # Averaged metrics
        # Only average over classes that have support (avoid NaN)
        has_support = self.support > 0
        
        if has_support.any():
            if self.average == 'macro':
                # Simple average over classes with support
                metrics['mean_precision'] = precision[has_support].mean()
                metrics['mean_recall'] = recall[has_support].mean()
                metrics['mean_f1'] = f1_score[has_support].mean()
                metrics['mean_iou'] = iou[has_support].mean()
            elif self.average == 'weighted':
                # Weighted average by class support
                weights = self.support[has_support].float()
                weights = weights / weights.sum()
                
                metrics['mean_precision'] = (precision[has_support] * weights).sum()
                metrics['mean_recall'] = (recall[has_support] * weights).sum()
                metrics['mean_f1'] = (f1_score[has_support] * weights).sum()
                metrics['mean_iou'] = (iou[has_support] * weights).sum()
        else:
            # No valid classes, return zeros
            metrics['mean_precision'] = torch.tensor(0.0)
            metrics['mean_recall'] = torch.tensor(0.0)
            metrics['mean_f1'] = torch.tensor(0.0)
            metrics['mean_iou'] = torch.tensor(0.0)
        
        return metrics
    
    def get_confusion_matrix(self) -> torch.Tensor:
        """
        Get the confusion matrix.
        
        Returns:
            torch.Tensor: Confusion matrix [num_classes, num_classes]
        """
        if not self.compute_confusion_matrix:
            raise ValueError("Confusion matrix computation is disabled")
        
        return self.confusion_matrix.clone()
    
    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            save_path (str, optional): Path to save the figure
            figsize (Tuple[int, int]): Figure size
        
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.compute_confusion_matrix:
            raise ValueError("Confusion matrix computation is disabled")
        
        cm = self.confusion_matrix.cpu().numpy()
        
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            str: Classification report string
        """
        metrics = self.compute()
        
        report_lines = [
            "Classification Report",
            "=" * 50,
            f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
        ]
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics.get(f'precision_{class_name}', 0.0)
            recall = metrics.get(f'recall_{class_name}', 0.0)
            f1 = metrics.get(f'f1_{class_name}', 0.0)
            support = self.support[i].item()
            
            report_lines.append(
                f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10.0f}"
            )
        
        # Add summary metrics
        report_lines.extend([
            "-" * 50,
            f"{'Overall Acc':<15} {metrics['accuracy']:<10.3f}",
            f"{'Mean Precision':<15} {metrics['mean_precision']:<10.3f}",
            f"{'Mean Recall':<15} {metrics['mean_recall']:<10.3f}",
            f"{'Mean F1':<15} {metrics['mean_f1']:<10.3f}",
            f"{'Mean IoU':<15} {metrics['mean_iou']:<10.3f}",
        ])
        
        return "\n".join(report_lines)
    
    def reset(self) -> None:
        """Reset all metric states."""
        self.true_positives.zero_()
        self.false_positives.zero_()
        self.false_negatives.zero_()
        self.support.zero_()
        self.total_correct.zero_()
        self.total_pixels.zero_()
        
        if self.compute_confusion_matrix:
            self.confusion_matrix.zero_()


class PhenologyMetrics(Metric):
    """
    Metrics for phenological stage prediction.
    
    Computes accuracy and consistency metrics for temporal phenology prediction.
    """
    
    def __init__(self, num_stages: int = 4, stage_names: Optional[List[str]] = None):
        super().__init__()
        
        self.num_stages = num_stages
        
        if stage_names is None:
            self.stage_names = ['sowing', 'vegetative', 'flowering', 'maturity']
        else:
            self.stage_names = stage_names
        
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("stage_correct", default=torch.zeros(num_stages), dist_reduce_fx="sum")
        self.add_state("stage_total", default=torch.zeros(num_stages), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update phenology metrics.
        
        Args:
            preds (torch.Tensor): Phenology predictions [B, num_stages] or [B]
            target (torch.Tensor): Ground truth stages [B]
        """
        if preds.dim() == 2:  # Probabilities or logits
            preds = torch.argmax(preds, dim=1)
        
        # Overall accuracy
        correct = (preds == target).sum()
        self.correct += correct
        self.total += len(preds)
        
        # Per-stage accuracy
        for stage_id in range(self.num_stages):
            stage_mask = (target == stage_id)
            if stage_mask.any():
                stage_correct = ((preds == target) & stage_mask).sum()
                self.stage_correct[stage_id] += stage_correct
                self.stage_total[stage_id] += stage_mask.sum()
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute phenology metrics."""
        metrics = {}
        
        # Overall accuracy
        metrics['phenology_accuracy'] = self.correct.float() / (self.total + 1e-8)
        
        # Per-stage accuracy
        stage_accuracies = self.stage_correct / (self.stage_total + 1e-8)
        for i, stage_name in enumerate(self.stage_names):
            metrics[f'phenology_accuracy_{stage_name}'] = stage_accuracies[i]
        
        # Mean stage accuracy
        valid_stages = self.stage_total > 0
        if valid_stages.any():
            metrics['phenology_mean_accuracy'] = stage_accuracies[valid_stages].mean()
        else:
            metrics['phenology_mean_accuracy'] = torch.tensor(0.0)
        
        return metrics


class MetricsCollection:
    """
    Collection of all metrics for comprehensive evaluation.
    
    This class manages multiple metric objects and provides
    unified interface for updating and computing all metrics.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        include_phenology: bool = True
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        # Initialize metrics
        self.segmentation_metrics = SegmentationMetrics(
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index
        )
        
        if include_phenology:
            self.phenology_metrics = PhenologyMetrics()
        else:
            self.phenology_metrics = None
        
        # Additional torchmetrics
        self.additional_metrics = {
            'jaccard': torchmetrics.JaccardIndex(
                task='multiclass',
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='macro'
            ),
            'accuracy': torchmetrics.Accuracy(
                task='multiclass',
                num_classes=num_classes,
                ignore_index=ignore_index,
                average='macro'
            ),
        }
    
    def update(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """
        Update all metrics with new outputs and targets.
        
        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs
            targets (Dict[str, torch.Tensor]): Ground truth targets
        """
        # Update segmentation metrics
        if 'segmentation' in outputs and 'labels' in targets:
            self.segmentation_metrics.update(outputs['segmentation'], targets['labels'])
            
            # Update additional metrics
            for metric in self.additional_metrics.values():
                metric.update(outputs['segmentation'], targets['labels'])
        
        # Update phenology metrics
        if (self.phenology_metrics is not None and 
            'phenology' in outputs and 'phenology_targets' in targets):
            self.phenology_metrics.update(outputs['phenology'], targets['phenology_targets'])
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics."""
        all_metrics = {}
        
        # Segmentation metrics
        seg_metrics = self.segmentation_metrics.compute()
        all_metrics.update(seg_metrics)
        
        # Phenology metrics
        if self.phenology_metrics is not None:
            phen_metrics = self.phenology_metrics.compute()
            all_metrics.update(phen_metrics)
        
        # Additional metrics
        for name, metric in self.additional_metrics.items():
            try:
                all_metrics[f'additional_{name}'] = metric.compute()
            except:
                # Handle cases where metric computation fails
                all_metrics[f'additional_{name}'] = torch.tensor(0.0)
        
        return all_metrics
    
    def reset(self):
        """Reset all metrics."""
        self.segmentation_metrics.reset()
        
        if self.phenology_metrics is not None:
            self.phenology_metrics.reset()
        
        for metric in self.additional_metrics.values():
            metric.reset()
    
    def get_summary(self) -> str:
        """Get summary of all metrics."""
        metrics = self.compute()
        
        summary_lines = [
            "Metrics Summary",
            "=" * 40,
            f"Overall Accuracy: {metrics.get('accuracy', 0.0):.4f}",
            f"Mean IoU: {metrics.get('mean_iou', 0.0):.4f}",
            f"Mean F1: {metrics.get('mean_f1', 0.0):.4f}",
        ]
        
        if 'phenology_accuracy' in metrics:
            summary_lines.extend([
                "",
                "Phenology Metrics:",
                f"Phenology Accuracy: {metrics['phenology_accuracy']:.4f}",
                f"Mean Stage Accuracy: {metrics.get('phenology_mean_accuracy', 0.0):.4f}",
            ])
        
        return "\n".join(summary_lines)


# Export for easy imports
__all__ = [
    'SegmentationMetrics',
    'PhenologyMetrics', 
    'MetricsCollection'
]

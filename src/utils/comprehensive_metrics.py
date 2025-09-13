"""
Comprehensive Metrics for Enhanced AMPT Model Evaluation

This module provides all requested evaluation metrics including:
- Loss score
- F1 score (macro and per-class)
- Jaccard index (IoU) (macro and per-class)
- IoU index micro
- Loss value tracking
- Accuracy of each crop class
- Jaccard index for each crop class

Core Innovation Metrics:
1. Phenological Attention Analysis
2. Hierarchical Scale Contribution
3. Foundation Model Adaptation Performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support,
    jaccard_score
)
import pandas as pd
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class EnhancedAMPTMetrics:
    """
    Comprehensive metrics collection for Enhanced AMPT model evaluation.
    
    Provides all requested metrics:
    - Loss scores and values
    - F1 scores (macro and per-class)
    - Jaccard indices (macro and per-class)
    - IoU micro and macro
    - Per-class accuracy
    - Innovation-specific metrics
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        device: Optional[torch.device] = None
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or torch.device('cpu')
        
        # Default crop class names for AgriFieldNet
        if class_names is None:
            self.class_names = ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Other']
        else:
            self.class_names = class_names
        
        self.reset()
        
        # Loss functions for different tasks
        self.crop_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.phenology_loss_fn = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized Enhanced AMPT Metrics for {num_classes} classes")
    
    def reset(self):
        """Reset all metric accumulators."""
        # Loss tracking
        self.losses = {
            'total_loss': [],
            'crop_loss': [],
            'segmentation_loss': [],
            'phenology_loss': [],
            'consistency_loss': []
        }
        
        # Prediction and target collections
        self.predictions = {
            'crop_preds': [],
            'crop_targets': [],
            'seg_preds': [],
            'seg_targets': [],
            'phenology_preds': [],
            'phenology_targets': []
        }
        
        # Innovation-specific metrics
        self.innovation_metrics = {
            'modal_weights': [],
            'phenological_stages': [],
            'scale_contributions': [],
            'attention_weights': []
        }
        
        logger.debug("Metrics reset successfully")
    
    def update_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        losses: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth targets dictionary
            losses: Computed losses dictionary (optional)
        """
        # Update losses if provided
        if losses is not None:
            for loss_name, loss_value in losses.items():
                if loss_name in self.losses:
                    self.losses[loss_name].append(loss_value.item())
        
        # Update crop classification predictions
        if 'crop_logits' in outputs and 'crop_labels' in targets:
            crop_preds = torch.softmax(outputs['crop_logits'], dim=1)
            crop_pred_classes = torch.argmax(crop_preds, dim=1)
            
            self.predictions['crop_preds'].extend(crop_pred_classes.cpu().numpy())
            self.predictions['crop_targets'].extend(targets['crop_labels'].cpu().numpy())
        
        # Update segmentation predictions
        if 'segmentation_logits' in outputs and 'mask' in targets:
            seg_preds = torch.softmax(outputs['segmentation_logits'], dim=1)
            seg_pred_classes = torch.argmax(seg_preds, dim=1)
            
            # Flatten and filter valid pixels
            seg_preds_flat = seg_pred_classes.flatten()
            seg_targets_flat = targets['mask'].flatten()
            valid_mask = seg_targets_flat != self.ignore_index
            
            if valid_mask.any():
                self.predictions['seg_preds'].extend(seg_preds_flat[valid_mask].cpu().numpy())
                self.predictions['seg_targets'].extend(seg_targets_flat[valid_mask].cpu().numpy())
        
        # Update phenology predictions
        if 'phenology_logits' in outputs and 'phenology_labels' in targets:
            pheno_preds = torch.softmax(outputs['phenology_logits'], dim=1)
            pheno_pred_classes = torch.argmax(pheno_preds, dim=1)
            
            self.predictions['phenology_preds'].extend(pheno_pred_classes.cpu().numpy())
            self.predictions['phenology_targets'].extend(targets['phenology_labels'].cpu().numpy())
        
        # Update innovation-specific metrics
        if 'modal_weights' in outputs:
            self.innovation_metrics['modal_weights'].extend(
                outputs['modal_weights'].cpu().numpy()
            )
        
        if 'stage_logits' in outputs:
            stage_probs = torch.softmax(outputs['stage_logits'], dim=1)
            self.innovation_metrics['phenological_stages'].extend(
                stage_probs.cpu().numpy()
            )
        
        if 'scale_features' in outputs and isinstance(outputs['scale_features'], dict):
            # Extract scale contribution information
            scale_output = outputs['scale_features']
            if all(key in scale_output for key in ['field_features', 'landscape_features', 'regional_features']):
                field_mag = torch.norm(scale_output['field_features'], dim=1).mean(dim=(1, 2))
                landscape_mag = torch.norm(scale_output['landscape_features'], dim=1).mean(dim=(1, 2))
                regional_mag = torch.norm(scale_output['regional_features'], dim=1).mean(dim=(1, 2))
                
                total_mag = field_mag + landscape_mag + regional_mag
                contributions = torch.stack([
                    field_mag / total_mag,
                    landscape_mag / total_mag,
                    regional_mag / total_mag
                ], dim=1)
                
                self.innovation_metrics['scale_contributions'].extend(
                    contributions.cpu().numpy()
                )
    
    def compute_loss_metrics(self) -> Dict[str, float]:
        """Compute loss-related metrics."""
        loss_metrics = {}
        
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                loss_metrics[f'{loss_name}_mean'] = np.mean(loss_values)
                loss_metrics[f'{loss_name}_std'] = np.std(loss_values)
                loss_metrics[f'{loss_name}_final'] = loss_values[-1]
        
        return loss_metrics
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        metrics = {}
        
        # Crop classification metrics
        if self.predictions['crop_preds'] and self.predictions['crop_targets']:
            crop_preds = np.array(self.predictions['crop_preds'])
            crop_targets = np.array(self.predictions['crop_targets'])
            
            # Overall accuracy
            metrics['crop_accuracy_overall'] = accuracy_score(crop_targets, crop_preds)
            
            # Per-class accuracy
            for i, class_name in enumerate(self.class_names):
                class_mask = crop_targets == i
                if class_mask.any():
                    class_acc = accuracy_score(
                        crop_targets[class_mask], 
                        crop_preds[class_mask]
                    )
                    metrics[f'crop_accuracy_{class_name}'] = class_acc
                else:
                    metrics[f'crop_accuracy_{class_name}'] = 0.0
            
            # Precision, Recall, F1 (macro and per-class)
            precision, recall, f1, support = precision_recall_fscore_support(
                crop_targets, crop_preds, average=None, zero_division=0
            )
            
            # Per-class metrics
            for i, class_name in enumerate(self.class_names):
                metrics[f'crop_precision_{class_name}'] = precision[i]
                metrics[f'crop_recall_{class_name}'] = recall[i]
                metrics[f'crop_f1_{class_name}'] = f1[i]
                metrics[f'crop_support_{class_name}'] = support[i]
            
            # Macro averages
            metrics['crop_precision_macro'] = np.mean(precision)
            metrics['crop_recall_macro'] = np.mean(recall)
            metrics['crop_f1_macro'] = np.mean(f1)
            
            # Weighted averages
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                crop_targets, crop_preds, average='weighted', zero_division=0
            )
            metrics['crop_precision_weighted'] = precision_w
            metrics['crop_recall_weighted'] = recall_w
            metrics['crop_f1_weighted'] = f1_w
            
            # Jaccard Index (IoU) per class and macro
            try:
                jaccard_per_class = jaccard_score(
                    crop_targets, crop_preds, average=None, zero_division=0
                )
                
                for i, class_name in enumerate(self.class_names):
                    metrics[f'crop_jaccard_{class_name}'] = jaccard_per_class[i]
                
                metrics['crop_jaccard_macro'] = np.mean(jaccard_per_class)
                
                # Micro average Jaccard
                metrics['crop_jaccard_micro'] = jaccard_score(
                    crop_targets, crop_preds, average='micro', zero_division=0
                )
                
                # Weighted average Jaccard
                metrics['crop_jaccard_weighted'] = jaccard_score(
                    crop_targets, crop_preds, average='weighted', zero_division=0
                )
                
            except Exception as e:
                logger.warning(f"Error computing Jaccard scores: {e}")
                for i, class_name in enumerate(self.class_names):
                    metrics[f'crop_jaccard_{class_name}'] = 0.0
                metrics['crop_jaccard_macro'] = 0.0
                metrics['crop_jaccard_micro'] = 0.0
                metrics['crop_jaccard_weighted'] = 0.0
        
        # Segmentation metrics (if available)
        if self.predictions['seg_preds'] and self.predictions['seg_targets']:
            seg_preds = np.array(self.predictions['seg_preds'])
            seg_targets = np.array(self.predictions['seg_targets'])
            
            # Overall segmentation accuracy
            metrics['seg_accuracy_overall'] = accuracy_score(seg_targets, seg_preds)
            
            # Per-class segmentation accuracy
            for i, class_name in enumerate(self.class_names):
                class_mask = seg_targets == i
                if class_mask.any():
                    class_acc = accuracy_score(
                        seg_targets[class_mask], 
                        seg_preds[class_mask]
                    )
                    metrics[f'seg_accuracy_{class_name}'] = class_acc
                else:
                    metrics[f'seg_accuracy_{class_name}'] = 0.0
            
            # Segmentation F1 scores
            seg_precision, seg_recall, seg_f1, seg_support = precision_recall_fscore_support(
                seg_targets, seg_preds, average=None, zero_division=0
            )
            
            for i, class_name in enumerate(self.class_names):
                metrics[f'seg_f1_{class_name}'] = seg_f1[i]
            
            metrics['seg_f1_macro'] = np.mean(seg_f1)
            
            # Segmentation Jaccard (IoU)
            try:
                seg_jaccard_per_class = jaccard_score(
                    seg_targets, seg_preds, average=None, zero_division=0
                )
                
                for i, class_name in enumerate(self.class_names):
                    metrics[f'seg_jaccard_{class_name}'] = seg_jaccard_per_class[i]
                
                metrics['seg_jaccard_macro'] = np.mean(seg_jaccard_per_class)
                metrics['seg_jaccard_micro'] = jaccard_score(
                    seg_targets, seg_preds, average='micro', zero_division=0
                )
                
            except Exception as e:
                logger.warning(f"Error computing segmentation Jaccard scores: {e}")
                for i, class_name in enumerate(self.class_names):
                    metrics[f'seg_jaccard_{class_name}'] = 0.0
                metrics['seg_jaccard_macro'] = 0.0
                metrics['seg_jaccard_micro'] = 0.0
        
        return metrics
    
    def compute_innovation_metrics(self) -> Dict[str, float]:
        """Compute metrics specific to the three core innovations."""
        metrics = {}
        
        # Core Innovation 1: Cross-Modal Phenological Attention
        if self.innovation_metrics['modal_weights']:
            modal_weights = np.array(self.innovation_metrics['modal_weights'])  # [N, 2]
            
            metrics['cmpa_sar_weight_mean'] = np.mean(modal_weights[:, 0])
            metrics['cmpa_sar_weight_std'] = np.std(modal_weights[:, 0])
            metrics['cmpa_optical_weight_mean'] = np.mean(modal_weights[:, 1])
            metrics['cmpa_optical_weight_std'] = np.std(modal_weights[:, 1])
            
            # Modal balance (how well balanced the attention is)
            modal_balance = 1 - np.abs(modal_weights[:, 0] - modal_weights[:, 1])
            metrics['cmpa_modal_balance_mean'] = np.mean(modal_balance)
            
            # Correlation between SAR and optical weights
            correlation = np.corrcoef(modal_weights[:, 0], modal_weights[:, 1])[0, 1]
            metrics['cmpa_modal_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Phenological stage distribution
        if self.innovation_metrics['phenological_stages']:
            pheno_stages = np.array(self.innovation_metrics['phenological_stages'])  # [N, 5]
            
            stage_names = ['Sowing', 'Vegetative', 'Flowering', 'Maturation', 'Harvest']
            for i, stage_name in enumerate(stage_names):
                metrics[f'cmpa_stage_prob_{stage_name.lower()}'] = np.mean(pheno_stages[:, i])
            
            # Stage entropy (diversity of phenological predictions)
            stage_entropy = -np.sum(pheno_stages * np.log(pheno_stages + 1e-8), axis=1)
            metrics['cmpa_stage_entropy_mean'] = np.mean(stage_entropy)
        
        # Core Innovation 2: Hierarchical Scale-Adaptive Fusion
        if self.innovation_metrics['scale_contributions']:
            scale_contribs = np.array(self.innovation_metrics['scale_contributions'])  # [N, 3]
            
            scale_names = ['Field', 'Landscape', 'Regional']
            for i, scale_name in enumerate(scale_names):
                metrics[f'hsaf_scale_contrib_{scale_name.lower()}'] = np.mean(scale_contribs[:, i])
                metrics[f'hsaf_scale_contrib_{scale_name.lower()}_std'] = np.std(scale_contribs[:, i])
            
            # Scale balance (how evenly distributed the contributions are)
            scale_balance = 1 - np.std(scale_contribs, axis=1)
            metrics['hsaf_scale_balance_mean'] = np.mean(scale_balance)
            
            # Dominant scale analysis
            dominant_scales = np.argmax(scale_contribs, axis=1)
            for i, scale_name in enumerate(scale_names):
                dominance_ratio = np.mean(dominant_scales == i)
                metrics[f'hsaf_dominance_{scale_name.lower()}'] = dominance_ratio
        
        return metrics
    
    def compute_comprehensive_metrics(self) -> Dict[str, Union[float, Dict]]:
        """Compute all metrics and return comprehensive results."""
        results = {
            'loss_metrics': self.compute_loss_metrics(),
            'classification_metrics': self.compute_classification_metrics(),
            'innovation_metrics': self.compute_innovation_metrics()
        }
        
        # Flatten for easy access
        flattened = {}
        for category, metrics in results.items():
            for metric_name, value in metrics.items():
                flattened[f"{category}_{metric_name}"] = value
        
        results['all_metrics'] = flattened
        
        return results
    
    def get_confusion_matrix(self, task: str = 'crop') -> np.ndarray:
        """Get confusion matrix for specified task."""
        if task == 'crop' and self.predictions['crop_preds']:
            return confusion_matrix(
                self.predictions['crop_targets'],
                self.predictions['crop_preds'],
                labels=list(range(self.num_classes))
            )
        elif task == 'segmentation' and self.predictions['seg_preds']:
            return confusion_matrix(
                self.predictions['seg_targets'],
                self.predictions['seg_preds'],
                labels=list(range(self.num_classes))
            )
        else:
            return np.zeros((self.num_classes, self.num_classes))
    
    def plot_confusion_matrix(
        self,
        task: str = 'crop',
        normalize: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix(task)
        
        if normalize and cm.sum() > 0:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
            title = f'Normalized Confusion Matrix - {task.title()}'
            fmt = '.2f'
        else:
            title = f'Confusion Matrix - {task.title()}'
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
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def generate_comprehensive_report(
        self,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        
        # Compute all metrics
        results = self.compute_comprehensive_metrics()
        
        # Create summary
        summary = {
            'model_name': 'Enhanced AMPT with Core Innovations',
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'total_samples': len(self.predictions['crop_targets']) if self.predictions['crop_targets'] else 0,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        # Key metrics summary
        key_metrics = {}
        if results['classification_metrics']:
            key_metrics.update({
                'Overall_Accuracy': results['classification_metrics'].get('crop_accuracy_overall', 0.0),
                'F1_Score_Macro': results['classification_metrics'].get('crop_f1_macro', 0.0),
                'Jaccard_Index_Macro': results['classification_metrics'].get('crop_jaccard_macro', 0.0),
                'Jaccard_Index_Micro': results['classification_metrics'].get('crop_jaccard_micro', 0.0),
            })
        
        if results['loss_metrics']:
            key_metrics.update({
                'Final_Loss': results['loss_metrics'].get('total_loss_final', 0.0),
                'Mean_Loss': results['loss_metrics'].get('total_loss_mean', 0.0),
            })
        
        summary['key_metrics'] = key_metrics
        
        # Per-class performance
        per_class_performance = {}
        for i, class_name in enumerate(self.class_names):
            per_class_performance[class_name] = {
                'accuracy': results['classification_metrics'].get(f'crop_accuracy_{class_name}', 0.0),
                'f1_score': results['classification_metrics'].get(f'crop_f1_{class_name}', 0.0),
                'jaccard_index': results['classification_metrics'].get(f'crop_jaccard_{class_name}', 0.0),
                'precision': results['classification_metrics'].get(f'crop_precision_{class_name}', 0.0),
                'recall': results['classification_metrics'].get(f'crop_recall_{class_name}', 0.0),
                'support': results['classification_metrics'].get(f'crop_support_{class_name}', 0)
            }
        
        summary['per_class_performance'] = per_class_performance
        
        # Innovation analysis
        innovation_analysis = {
            'cross_modal_phenological_attention': {
                'sar_attention_mean': results['innovation_metrics'].get('cmpa_sar_weight_mean', 0.0),
                'optical_attention_mean': results['innovation_metrics'].get('cmpa_optical_weight_mean', 0.0),
                'modal_balance': results['innovation_metrics'].get('cmpa_modal_balance_mean', 0.0),
                'modal_correlation': results['innovation_metrics'].get('cmpa_modal_correlation', 0.0),
            },
            'hierarchical_scale_adaptive_fusion': {
                'field_contribution': results['innovation_metrics'].get('hsaf_scale_contrib_field', 0.0),
                'landscape_contribution': results['innovation_metrics'].get('hsaf_scale_contrib_landscape', 0.0),
                'regional_contribution': results['innovation_metrics'].get('hsaf_scale_contrib_regional', 0.0),
                'scale_balance': results['innovation_metrics'].get('hsaf_scale_balance_mean', 0.0),
            }
        }
        
        summary['innovation_analysis'] = innovation_analysis
        
        # Complete report
        comprehensive_report = {
            'summary': summary,
            'detailed_results': results,
            'confusion_matrices': {
                'crop_classification': self.get_confusion_matrix('crop').tolist(),
                'segmentation': self.get_confusion_matrix('segmentation').tolist()
            }
        }
        
        # Save report if directory provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON report
            report_file = save_path / 'comprehensive_evaluation_report.json'
            with open(report_file, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            # Save confusion matrices
            self.plot_confusion_matrix('crop', True, str(save_path / 'confusion_matrix_crop.png'))
            if self.predictions['seg_preds']:
                self.plot_confusion_matrix('segmentation', True, str(save_path / 'confusion_matrix_segmentation.png'))
            
            # Save metrics CSV
            metrics_df = pd.DataFrame([results['all_metrics']])
            metrics_df.to_csv(save_path / 'detailed_metrics.csv', index=False)
            
            logger.info(f"Comprehensive report saved to {save_dir}")
        
        return comprehensive_report
    
    def print_summary(self):
        """Print a formatted summary of key metrics."""
        results = self.compute_comprehensive_metrics()
        
        print("\n" + "="*80)
        print("🌾 ENHANCED AMPT MODEL - COMPREHENSIVE EVALUATION SUMMARY 🌾")
        print("="*80)
        
        # Loss metrics
        if results['loss_metrics']:
            print("\n📊 LOSS METRICS:")
            for loss_name, value in results['loss_metrics'].items():
                if 'mean' in loss_name:
                    print(f"   • {loss_name.replace('_', ' ').title()}: {value:.6f}")
        
        # Classification performance
        if results['classification_metrics']:
            print("\n🎯 CLASSIFICATION PERFORMANCE:")
            metrics = results['classification_metrics']
            print(f"   • Overall Accuracy: {metrics.get('crop_accuracy_overall', 0.0):.4f}")
            print(f"   • F1 Score (Macro): {metrics.get('crop_f1_macro', 0.0):.4f}")
            print(f"   • F1 Score (Weighted): {metrics.get('crop_f1_weighted', 0.0):.4f}")
            print(f"   • Jaccard Index (Macro): {metrics.get('crop_jaccard_macro', 0.0):.4f}")
            print(f"   • Jaccard Index (Micro): {metrics.get('crop_jaccard_micro', 0.0):.4f}")
            print(f"   • Jaccard Index (Weighted): {metrics.get('crop_jaccard_weighted', 0.0):.4f}")
        
        # Per-class performance
        print("\n🌱 PER-CLASS PERFORMANCE:")
        for i, class_name in enumerate(self.class_names):
            if results['classification_metrics']:
                metrics = results['classification_metrics']
                acc = metrics.get(f'crop_accuracy_{class_name}', 0.0)
                f1 = metrics.get(f'crop_f1_{class_name}', 0.0)
                jaccard = metrics.get(f'crop_jaccard_{class_name}', 0.0)
                print(f"   • {class_name:12}: Acc={acc:.3f}, F1={f1:.3f}, IoU={jaccard:.3f}")
        
        # Innovation metrics
        if results['innovation_metrics']:
            print("\n🚀 CORE INNOVATIONS ANALYSIS:")
            
            print("   1️⃣ Cross-Modal Phenological Attention (CMPA):")
            inn_metrics = results['innovation_metrics']
            sar_weight = inn_metrics.get('cmpa_sar_weight_mean', 0.0)
            opt_weight = inn_metrics.get('cmpa_optical_weight_mean', 0.0)
            balance = inn_metrics.get('cmpa_modal_balance_mean', 0.0)
            print(f"      • SAR Attention Weight: {sar_weight:.3f}")
            print(f"      • Optical Attention Weight: {opt_weight:.3f}")
            print(f"      • Modal Balance: {balance:.3f}")
            
            print("   2️⃣ Hierarchical Scale-Adaptive Fusion (HSAF):")
            field_contrib = inn_metrics.get('hsaf_scale_contrib_field', 0.0)
            landscape_contrib = inn_metrics.get('hsaf_scale_contrib_landscape', 0.0)
            regional_contrib = inn_metrics.get('hsaf_scale_contrib_regional', 0.0)
            print(f"      • Field Scale Contribution: {field_contrib:.3f}")
            print(f"      • Landscape Scale Contribution: {landscape_contrib:.3f}")
            print(f"      • Regional Scale Contribution: {regional_contrib:.3f}")
        
        print("\n" + "="*80)

# Export
__all__ = ['EnhancedAMPTMetrics']

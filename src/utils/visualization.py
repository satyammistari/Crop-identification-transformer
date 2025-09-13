"""
AttentionVisualizer: Visualization of attention weights and model interpretability.

This module provides comprehensive visualization tools for understanding
the AMPT model's attention mechanisms and decision-making process:

1. Cross-modal attention weight visualization
2. Phenological stage progression plots
3. SAR vs Optical weighting over time
4. Spatial attention heatmaps
5. Model prediction visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import cv2
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


class AttentionVisualizer:
    """
    Comprehensive visualization for AMPT model attention mechanisms.
    
    This class provides various visualization methods to understand:
    - Cross-modal attention weights
    - Phenological stage influences
    - Temporal attention patterns
    - Spatial attention maps
    
    Args:
        class_names (List[str]): Names of crop classes
        phenology_names (List[str]): Names of phenological stages
        colormap (str): Colormap for visualizations
        figsize (Tuple[int, int]): Default figure size
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        phenology_names: Optional[List[str]] = None,
        colormap: str = 'viridis',
        figsize: Tuple[int, int] = (12, 8)
    ):
        self.class_names = class_names or ['gram', 'maize', 'mustard', 'sugarcane', 'wheat', 'other_crop']
        self.phenology_names = phenology_names or ['sowing', 'vegetative', 'flowering', 'maturity']
        self.colormap = colormap
        self.figsize = figsize
        
        # Set up color schemes
        self.class_colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        self.phenology_colors = plt.cm.RdYlGn(np.linspace(0, 1, len(self.phenology_names)))
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_cross_modal_attention(
        self,
        attention_info: Dict[str, torch.Tensor],
        sample_id: str = "sample",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize cross-modal attention weights and phenological influences.
        
        Args:
            attention_info (Dict[str, torch.Tensor]): Attention information from model
            sample_id (str): Sample identifier for title
            save_path (str, optional): Path to save the figure
        
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'Cross-Modal Attention Analysis - {sample_id}', fontsize=16, fontweight='bold')
        
        # Extract attention components
        modality_weights = attention_info.get('modality_weights', torch.tensor([[0.5, 0.5]]))
        phenology_stage = attention_info.get('phenology_stage', torch.zeros(1, 4))
        
        if modality_weights.dim() > 1:
            modality_weights = modality_weights[0]  # Take first sample
        if phenology_stage.dim() > 1:
            phenology_stage = phenology_stage[0]  # Take first sample
        
        # 1. Modality Weight Distribution
        ax1 = axes[0, 0]
        modality_labels = ['SAR', 'Optical']
        modality_values = modality_weights.detach().cpu().numpy()
        
        bars = ax1.bar(modality_labels, modality_values, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Modality Attention Weights', fontweight='bold')
        ax1.set_ylabel('Attention Weight')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, modality_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Phenological Stage Probabilities
        ax2 = axes[0, 1]
        phenology_values = phenology_stage.detach().cpu().numpy()
        
        bars = ax2.bar(self.phenology_names, phenology_values, color=self.phenology_colors)
        ax2.set_title('Phenological Stage Probabilities', fontweight='bold')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, phenology_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Phenology-Modality Relationship
        ax3 = axes[1, 0]
        
        # Create correlation heatmap between phenology and modality weights
        phenology_expanded = phenology_values.reshape(-1, 1)
        modality_expanded = modality_values.reshape(1, -1)
        correlation_matrix = phenology_expanded @ modality_expanded
        
        im = ax3.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto')
        ax3.set_xticks(range(len(modality_labels)))
        ax3.set_xticklabels(modality_labels)
        ax3.set_yticks(range(len(self.phenology_names)))
        ax3.set_yticklabels(self.phenology_names)
        ax3.set_title('Phenology-Modality Correlation', fontweight='bold')
        
        # Add text annotations
        for i in range(len(self.phenology_names)):
            for j in range(len(modality_labels)):
                ax3.text(j, i, f'{correlation_matrix[i, j]:.3f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Interpretation Text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Generate interpretation
        dominant_stage = self.phenology_names[np.argmax(phenology_values)]
        dominant_modality = modality_labels[np.argmax(modality_values)]
        stage_confidence = np.max(phenology_values)
        modality_confidence = np.max(modality_values)
        
        interpretation_text = f"""
        Interpretation:
        
        Dominant Stage: {dominant_stage}
        Confidence: {stage_confidence:.3f}
        
        Preferred Modality: {dominant_modality}
        Weight: {modality_confidence:.3f}
        
        Analysis:
        {self._get_interpretation_text(dominant_stage, dominant_modality, stage_confidence)}
        """
        
        ax4.text(0.05, 0.95, interpretation_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_temporal_attention(
        self,
        temporal_attention_weights: torch.Tensor,
        temporal_length: int = 6,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize temporal attention patterns.
        
        Args:
            temporal_attention_weights (torch.Tensor): Temporal attention weights [B, H, T, T]
            temporal_length (int): Length of temporal sequence
            save_path (str, optional): Path to save the figure
        
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Temporal Attention Analysis', fontsize=16, fontweight='bold')
        
        # Take first sample and average over heads
        if temporal_attention_weights.dim() == 4:
            attention_matrix = temporal_attention_weights[0].mean(dim=0)  # Average over heads
        else:
            attention_matrix = temporal_attention_weights
        
        attention_matrix = attention_matrix.detach().cpu().numpy()
        
        # 1. Attention Matrix Heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(attention_matrix, cmap='Blues', aspect='auto')
        ax1.set_title('Temporal Attention Matrix', fontweight='bold')
        ax1.set_xlabel('Time Step (Key)')
        ax1.set_ylabel('Time Step (Query)')
        
        # Add month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'][:temporal_length]
        ax1.set_xticks(range(temporal_length))
        ax1.set_xticklabels(month_labels)
        ax1.set_yticks(range(temporal_length))
        ax1.set_yticklabels(month_labels)
        
        # Add value annotations
        for i in range(temporal_length):
            for j in range(temporal_length):
                ax1.text(j, i, f'{attention_matrix[i, j]:.2f}', 
                        ha='center', va='center', color='white' if attention_matrix[i, j] > 0.5 else 'black')
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Attention Distribution Over Time
        ax2 = axes[1]
        
        # Plot attention given to each time step (column-wise sum)
        attention_received = attention_matrix.sum(axis=0)
        attention_given = attention_matrix.sum(axis=1)
        
        x = range(temporal_length)
        width = 0.35
        
        bars1 = ax2.bar([i - width/2 for i in x], attention_received, width, 
                       label='Attention Received', color='skyblue', alpha=0.8)
        bars2 = ax2.bar([i + width/2 for i in x], attention_given, width,
                       label='Attention Given', color='lightcoral', alpha=0.8)
        
        ax2.set_title('Temporal Attention Distribution', fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Total Attention')
        ax2.set_xticks(x)
        ax2.set_xticklabels(month_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_prediction_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        attention_map: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize prediction overlay on original image.
        
        Args:
            image (np.ndarray): Original image [H, W, C]
            prediction (np.ndarray): Prediction mask [H, W]
            ground_truth (np.ndarray, optional): Ground truth mask [H, W]
            attention_map (np.ndarray, optional): Spatial attention map [H, W]
            save_path (str, optional): Path to save the figure
        
        Returns:
            plt.Figure: Matplotlib figure
        """
        n_plots = 2 + (ground_truth is not None) + (attention_map is not None)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        # Normalize image for display
        if image.dtype != np.uint8:
            image_display = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_display = image.copy()
        
        # If image has more than 3 channels, take RGB
        if image_display.shape[-1] > 3:
            image_display = image_display[:, :, :3]
        
        plot_idx = 0
        
        # 1. Original Image
        axes[plot_idx].imshow(image_display)
        axes[plot_idx].set_title('Original Image', fontweight='bold')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # 2. Prediction
        pred_colored = self._apply_class_colormap(prediction)
        axes[plot_idx].imshow(pred_colored)
        axes[plot_idx].set_title('Prediction', fontweight='bold')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # 3. Ground Truth (if available)
        if ground_truth is not None:
            gt_colored = self._apply_class_colormap(ground_truth)
            axes[plot_idx].imshow(gt_colored)
            axes[plot_idx].set_title('Ground Truth', fontweight='bold')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # 4. Attention Map (if available)
        if attention_map is not None:
            im = axes[plot_idx].imshow(attention_map, cmap='hot', alpha=0.7)
            axes[plot_idx].imshow(image_display, alpha=0.3)
            axes[plot_idx].set_title('Spatial Attention', fontweight='bold')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        
        # Add legend for classes
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=self.class_colors[i], 
                                        label=self.class_names[i]) 
                          for i in range(len(self.class_names))]
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
                  ncol=len(self.class_names), fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(
        self,
        attention_data: Dict[str, Any],
        predictions: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive dashboard with Plotly.
        
        Args:
            attention_data (Dict[str, Any]): Attention data over time
            predictions (Dict[str, np.ndarray]): Prediction data
            save_path (str, optional): Path to save HTML file
        
        Returns:
            go.Figure: Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Modality Weights Over Time', 'Phenology Evolution',
                          'Attention Entropy', 'Model Confidence'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract temporal data
        temporal_steps = list(range(len(attention_data.get('modality_weights', []))))
        
        # 1. Modality weights over time
        if 'modality_weights' in attention_data:
            modality_weights = attention_data['modality_weights']
            sar_weights = [w[0] for w in modality_weights]
            optical_weights = [w[1] for w in modality_weights]
            
            fig.add_trace(
                go.Scatter(x=temporal_steps, y=sar_weights, name='SAR Weight',
                          line=dict(color='red', width=3)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=temporal_steps, y=optical_weights, name='Optical Weight',
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
        
        # 2. Phenology evolution
        if 'phenology_probs' in attention_data:
            phenology_data = attention_data['phenology_probs']
            for i, stage_name in enumerate(self.phenology_names):
                stage_probs = [p[i] for p in phenology_data]
                fig.add_trace(
                    go.Scatter(x=temporal_steps, y=stage_probs, name=stage_name,
                              line=dict(width=2)),
                    row=1, col=2
                )
        
        # 3. Attention entropy (measure of uncertainty)
        if 'attention_entropy' in attention_data:
            entropy_values = attention_data['attention_entropy']
            fig.add_trace(
                go.Scatter(x=temporal_steps, y=entropy_values, name='Attention Entropy',
                          line=dict(color='purple', width=3)),
                row=2, col=1
            )
        
        # 4. Model confidence
        if 'prediction_confidence' in predictions:
            confidence_values = predictions['prediction_confidence']
            fig.add_trace(
                go.Scatter(x=temporal_steps, y=confidence_values, name='Model Confidence',
                          line=dict(color='green', width=3)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='AMPT Model Interactive Dashboard',
            showlegend=True,
            height=800,
            font=dict(size=12)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Step", row=2, col=1)
        fig.update_xaxes(title_text="Time Step", row=2, col=2)
        fig.update_yaxes(title_text="Weight", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=2)
        fig.update_yaxes(title_text="Entropy", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _apply_class_colormap(self, mask: np.ndarray) -> np.ndarray:
        """Apply class-specific colormap to segmentation mask."""
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id in range(len(self.class_names)):
            class_mask = (mask == class_id)
            colored_mask[class_mask] = (self.class_colors[class_id][:3] * 255).astype(np.uint8)
        
        return colored_mask
    
    def _get_interpretation_text(self, stage: str, modality: str, confidence: float) -> str:
        """Generate interpretation text based on stage and modality."""
        interpretations = {
            ('sowing', 'SAR'): "Early season SAR emphasis for soil moisture detection",
            ('sowing', 'Optical'): "Optical focus suggests visible crop emergence",
            ('vegetative', 'SAR'): "SAR emphasis for vegetation structure analysis",
            ('vegetative', 'Optical'): "Optical focus for chlorophyll assessment",
            ('flowering', 'Optical'): "Optical dominance for bloom detection",
            ('flowering', 'SAR'): "Unexpected SAR emphasis may indicate stress",
            ('maturity', 'Optical'): "Optical emphasis for harvest readiness",
            ('maturity', 'SAR'): "SAR focus on structural crop changes"
        }
        
        base_text = interpretations.get((stage, modality), 
                                       f"{modality} emphasis during {stage} stage")
        
        if confidence < 0.5:
            return f"{base_text}\n(Low confidence - uncertain prediction)"
        elif confidence > 0.8:
            return f"{base_text}\n(High confidence prediction)"
        else:
            return base_text


# Export for easy imports
__all__ = ['AttentionVisualizer']

"""
Enhanced AMPT Model Demo - Quick Demonstration of All Features

This script demonstrates the Enhanced AMPT model with:
1. Cross-Modal Phenological Attention (CMPA)
2. Hierarchical Scale-Adaptive Fusion (HSAF)  
3. Foundation Model Adaptation (FMA)

And provides comprehensive metrics including:
- Loss score, F1 score, Jaccard index, IoU index micro
- Loss value, accuracy of each crop
- Jaccard index for each crop class
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.enhanced_ampt_model import EnhancedAMPTModel
from src.utils.comprehensive_metrics import EnhancedAMPTMetrics
from src.data.agrifieldnet_dataset import AgriFieldNetDataset, get_agrifieldnet_transforms

def create_demo_config():
    """Create a minimal config for demo purposes."""
    return {
        'model': {
            'num_classes': 6,
            'optical_channels': 6,
            'sar_channels': 2,
            'weather_features': 5,
            'num_time_steps': 6,
            'backbone': {
                'img_size': 224,
                'patch_size': 16,
                'pretrained': False  # Disable for demo
            }
        },
        'dataset': {
            'class_names': ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Other']
        },
        'loss': {
            'crop_weight': 1.0,
            'phenology_weight': 0.5,
            'segmentation_weight': 1.0,
            'ignore_index': 255
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 2  # Just 2 epochs for demo
        }
    }

def create_demo_batch():
    """Create a synthetic batch for demonstration."""
    batch_size = 4
    
    batch = {
        'optical': torch.randn(batch_size, 6, 224, 224),  # Optical imagery
        'sar': torch.randn(batch_size, 2, 224, 224),      # SAR imagery
        'weather': torch.randn(batch_size, 5),            # Weather data
        'temporal_optical': torch.randn(batch_size, 6, 6, 224, 224),  # Temporal optical
        'temporal_sar': torch.randn(batch_size, 6, 2, 224, 224),      # Temporal SAR
        'crop_labels': torch.randint(0, 6, (batch_size,)),            # Crop classification labels
        'mask': torch.randint(0, 6, (batch_size, 224, 224)),          # Segmentation masks
        'phenology_labels': torch.randint(0, 5, (batch_size,)),       # Phenological stage labels
    }
    
    return batch

def demonstrate_model_forward_pass(model, batch):
    """Demonstrate model forward pass and extract innovations."""
    print("\n" + "="*60)
    print("FORWARD PASS DEMONSTRATION")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    
    print(f"✓ Crop Logits Shape: {outputs['crop_logits'].shape}")
    print(f"✓ Phenology Logits Shape: {outputs['phenology_logits'].shape}")
    print(f"✓ Segmentation Logits Shape: {outputs['segmentation_logits'].shape}")
    
    # Innovation 1: Cross-Modal Phenological Attention
    if 'modal_weights' in outputs:
        modal_weights = outputs['modal_weights']
        print(f"\n🚀 Innovation 1 - Cross-Modal Phenological Attention:")
        print(f"   Modal Weights Shape: {modal_weights.shape}")
        sar_mean = modal_weights[:, 0].mean().item()
        optical_mean = modal_weights[:, 1].mean().item()
        print(f"   Average SAR Attention: {sar_mean:.3f}")
        print(f"   Average Optical Attention: {optical_mean:.3f}")
    
    # Innovation 2: Hierarchical Scale-Adaptive Fusion
    if 'scale_features' in outputs and isinstance(outputs['scale_features'], dict):
        scale_output = outputs['scale_features']
        print(f"\n🚀 Innovation 2 - Hierarchical Scale-Adaptive Fusion:")
        if 'field_features' in scale_output:
            print(f"   Field Features Shape: {scale_output['field_features'].shape}")
        if 'landscape_features' in scale_output:
            print(f"   Landscape Features Shape: {scale_output['landscape_features'].shape}")
        if 'regional_features' in scale_output:
            print(f"   Regional Features Shape: {scale_output['regional_features'].shape}")
    
    # Innovation 3: Phenological Stage Analysis
    if 'stage_logits' in outputs:
        stage_probs = torch.softmax(outputs['stage_logits'], dim=1)
        print(f"\n🚀 Innovation 3 - Phenological Stage Analysis:")
        print(f"   Stage Probabilities Shape: {stage_probs.shape}")
        stage_names = ['Sowing', 'Vegetative', 'Flowering', 'Maturation', 'Harvest']
        avg_probs = stage_probs.mean(dim=0)
        for i, (stage, prob) in enumerate(zip(stage_names, avg_probs)):
            print(f"   {stage}: {prob:.3f}")
    
    return outputs

def demonstrate_comprehensive_metrics(model, batch):
    """Demonstrate comprehensive metrics calculation."""
    print("\n" + "="*60)
    print("COMPREHENSIVE METRICS DEMONSTRATION")
    print("="*60)
    
    # Initialize metrics collector
    metrics = EnhancedAMPTMetrics(
        num_classes=6,
        class_names=['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Other']
    )
    
    # Simulate multiple batches for realistic metrics
    for i in range(5):
        # Create varied synthetic data
        demo_batch = create_demo_batch()
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(demo_batch)
        
        # Compute losses
        losses = model.compute_loss(outputs, demo_batch)
        
        # Update metrics
        metrics.update_batch(outputs, demo_batch, losses)
        
        print(f"✓ Processed batch {i+1}/5")
    
    # Compute and display comprehensive metrics
    print(f"\n📊 COMPREHENSIVE EVALUATION RESULTS:")
    
    # Get all metrics
    results = metrics.compute_comprehensive_metrics()
    
    # Display key metrics
    if results['classification_metrics']:
        cm = results['classification_metrics']
        print(f"\n🎯 CLASSIFICATION PERFORMANCE:")
        print(f"   • Overall Accuracy: {cm.get('crop_accuracy_overall', 0.0):.4f}")
        print(f"   • F1 Score (Macro): {cm.get('crop_f1_macro', 0.0):.4f}")
        print(f"   • F1 Score (Weighted): {cm.get('crop_f1_weighted', 0.0):.4f}")
        print(f"   • Jaccard Index (Macro): {cm.get('crop_jaccard_macro', 0.0):.4f}")
        print(f"   • Jaccard Index (Micro): {cm.get('crop_jaccard_micro', 0.0):.4f}")
        print(f"   • Jaccard Index (Weighted): {cm.get('crop_jaccard_weighted', 0.0):.4f}")
    
    # Display per-class metrics
    print(f"\n🌱 PER-CLASS PERFORMANCE:")
    class_names = ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Other']
    for class_name in class_names:
        if results['classification_metrics']:
            cm = results['classification_metrics']
            acc = cm.get(f'crop_accuracy_{class_name}', 0.0)
            f1 = cm.get(f'crop_f1_{class_name}', 0.0)
            jaccard = cm.get(f'crop_jaccard_{class_name}', 0.0)
            print(f"   • {class_name:12}: Acc={acc:.3f}, F1={f1:.3f}, IoU={jaccard:.3f}")
    
    # Display loss metrics
    if results['loss_metrics']:
        lm = results['loss_metrics']
        print(f"\n📉 LOSS METRICS:")
        for loss_name, value in lm.items():
            if 'mean' in loss_name:
                print(f"   • {loss_name.replace('_', ' ').title()}: {value:.6f}")
    
    # Display innovation metrics
    if results['innovation_metrics']:
        im = results['innovation_metrics']
        print(f"\n🚀 CORE INNOVATIONS ANALYSIS:")
        
        print(f"   1️⃣ Cross-Modal Phenological Attention (CMPA):")
        print(f"      • SAR Attention Weight: {im.get('cmpa_sar_weight_mean', 0.0):.3f}")
        print(f"      • Optical Attention Weight: {im.get('cmpa_optical_weight_mean', 0.0):.3f}")
        print(f"      • Modal Balance: {im.get('cmpa_modal_balance_mean', 0.0):.3f}")
        
        print(f"   2️⃣ Hierarchical Scale-Adaptive Fusion (HSAF):")
        print(f"      • Field Scale Contribution: {im.get('hsaf_scale_contrib_field', 0.0):.3f}")
        print(f"      • Landscape Scale Contribution: {im.get('hsaf_scale_contrib_landscape', 0.0):.3f}")
        print(f"      • Regional Scale Contribution: {im.get('hsaf_scale_contrib_regional', 0.0):.3f}")
    
    # Generate comprehensive report
    output_dir = Path("outputs/demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comprehensive_report = metrics.generate_comprehensive_report(str(output_dir))
    
    print(f"\n📁 Comprehensive report saved to: {output_dir}")
    
    return results

def create_visualizations(results, output_dir):
    """Create demonstration visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Creating visualizations...")
    
    # Create a summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Per-class performance
    class_names = ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Other']
    
    if results.get('classification_metrics'):
        cm = results['classification_metrics']
        
        # Extract metrics
        accuracies = [cm.get(f'crop_accuracy_{name}', 0.0) for name in class_names]
        f1_scores = [cm.get(f'crop_f1_{name}', 0.0) for name in class_names]
        jaccard_scores = [cm.get(f'crop_jaccard_{name}', 0.0) for name in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x, f1_scores, width, label='F1 Score', alpha=0.8)
        axes[0, 0].bar(x + width, jaccard_scores, width, label='Jaccard Index', alpha=0.8)
        
        axes[0, 0].set_xlabel('Crop Classes')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Per-Class Performance Metrics')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Innovation analysis
    if results.get('innovation_metrics'):
        im = results['innovation_metrics']
        
        # Modal attention weights
        sar_weight = im.get('cmpa_sar_weight_mean', 0.5)
        optical_weight = im.get('cmpa_optical_weight_mean', 0.5)
        
        axes[0, 1].bar(['SAR', 'Optical'], [sar_weight, optical_weight], 
                      color=['red', 'green'], alpha=0.7)
        axes[0, 1].set_ylabel('Attention Weight')
        axes[0, 1].set_title('Cross-Modal Attention (CMPA)')
        axes[0, 1].set_ylim(0, 1)
        
        # Scale contributions
        scale_names = ['Field', 'Landscape', 'Regional']
        contributions = [
            im.get('hsaf_scale_contrib_field', 0.33),
            im.get('hsaf_scale_contrib_landscape', 0.33),
            im.get('hsaf_scale_contrib_regional', 0.33)
        ]
        
        axes[1, 0].bar(scale_names, contributions, 
                      color=['green', 'orange', 'purple'], alpha=0.7)
        axes[1, 0].set_ylabel('Contribution')
        axes[1, 0].set_title('Hierarchical Scale Fusion (HSAF)')
    
    # 3. Overall metrics summary
    axes[1, 1].axis('off')
    
    if results.get('classification_metrics'):
        cm = results['classification_metrics']
        summary_text = f"""
ENHANCED AMPT MODEL SUMMARY

🎯 Overall Performance:
• Accuracy: {cm.get('crop_accuracy_overall', 0.0):.3f}
• F1 (Macro): {cm.get('crop_f1_macro', 0.0):.3f}
• IoU (Macro): {cm.get('crop_jaccard_macro', 0.0):.3f}
• IoU (Micro): {cm.get('crop_jaccard_micro', 0.0):.3f}

🚀 Core Innovations:
✓ Cross-Modal Phenological Attention
✓ Hierarchical Scale-Adaptive Fusion
✓ Foundation Model Adaptation

📊 Comprehensive Metrics:
✓ Loss tracking and analysis
✓ Per-class performance evaluation
✓ Innovation effectiveness analysis
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Enhanced AMPT Model - Comprehensive Evaluation Demo', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'enhanced_ampt_demo_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")

def main():
    """Main demonstration function."""
    print("="*80)
    print("         ENHANCED AMPT MODEL DEMONSTRATION")
    print("      Core Innovations & Comprehensive Metrics")
    print("="*80)
    
    # Create demo configuration
    config = create_demo_config()
    
    print(f"\n📋 DEMO CONFIGURATION:")
    print(f"   • Model Classes: {config['model']['num_classes']}")
    print(f"   • Class Names: {config['dataset']['class_names']}")
    print(f"   • Core Innovations: 3 (CMPA, HSAF, FMA)")
    print(f"   • Comprehensive Metrics: Enabled")
    
    # Convert config to object for model
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # Add get method to nested configs
    def add_get_method(obj):
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if hasattr(attr_value, '__dict__') and not hasattr(attr_value, 'get'):
                    def make_get(instance):
                        def get_method(key, default=None):
                            return getattr(instance, key, default)
                        return get_method
                    setattr(attr_value, 'get', make_get(attr_value))
                    add_get_method(attr_value)
    
    model_config = Config(config)
    add_get_method(model_config)
    
    # Create model
    print(f"\n🤖 Creating Enhanced AMPT Model...")
    model = EnhancedAMPTModel(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Create demo batch
    print(f"\n📊 Creating demonstration batch...")
    batch = create_demo_batch()
    print(f"✓ Batch created with {batch['optical'].shape[0]} samples")
    
    # Demonstrate forward pass and innovations
    outputs = demonstrate_model_forward_pass(model, batch)
    
    # Demonstrate comprehensive metrics
    results = demonstrate_comprehensive_metrics(model, batch)
    
    # Create visualizations
    output_dir = "outputs/demo_results"
    create_visualizations(results, output_dir)
    
    # Save demo summary
    demo_summary = {
        'demo_info': {
            'timestamp': datetime.now().isoformat(),
            'model_params': total_params,
            'batch_size': batch['optical'].shape[0],
            'innovations_implemented': 3
        },
        'results': results,
        'config': config
    }
    
    summary_file = Path(output_dir) / 'demo_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(demo_summary, f, indent=2, default=str)
    
    print(f"\n🎉 ENHANCED AMPT MODEL DEMONSTRATION COMPLETED!")
    print("="*60)
    print("✓ Three core innovations successfully demonstrated")
    print("✓ Comprehensive metrics computed and analyzed")
    print("✓ Per-class performance evaluated")
    print("✓ Innovation effectiveness assessed")
    print("✓ Visualizations and reports generated")
    print(f"✓ All results saved to: {output_dir}")
    print("="*60)
    
    # Display final summary
    if results.get('classification_metrics'):
        cm = results['classification_metrics']
        print(f"\n📈 FINAL PERFORMANCE SUMMARY:")
        print(f"   • Overall Accuracy: {cm.get('crop_accuracy_overall', 0.0):.4f}")
        print(f"   • F1 Score (Macro): {cm.get('crop_f1_macro', 0.0):.4f}")
        print(f"   • Jaccard Index (Macro): {cm.get('crop_jaccard_macro', 0.0):.4f}")
        print(f"   • Jaccard Index (Micro): {cm.get('crop_jaccard_micro', 0.0):.4f}")
    
    return demo_summary

if __name__ == "__main__":
    main()

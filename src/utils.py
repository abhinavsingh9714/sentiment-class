"""
Utility functions for RNN sentiment classification project.
"""
import torch
import numpy as np
import random
import platform
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Disable nondeterministic operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_hardware_info():
    """Get hardware information for reporting."""
    info = {
        'platform': platform.system(),
        'processor': platform.processor(),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_name'] = torch.cuda.get_device_name(0)
    else:
        info['cuda_version'] = None
        info['cudnn_version'] = None
        info['gpu_name'] = None
    
    return info


def log_hardware_info():
    """Log hardware information to console."""
    info = get_hardware_info()
    print("\n" + "="*50)
    print("HARDWARE INFORMATION")
    print("="*50)
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"RAM: {info['ram_gb']} GB")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"GPU: {info['gpu_name']}")
    else:
        print("Using CPU only")
    print("="*50 + "\n")


def save_metrics_to_csv(metrics_list, output_path='results/metrics.csv'):
    """Save metrics to CSV file."""
    df = pd.DataFrame(metrics_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def plot_accuracy_f1_vs_seqlen(metrics_df, output_path='results/plots/accuracy_f1_vs_seqlen.png'):
    """Plot Accuracy and F1-score vs Sequence Length."""
    # Filter rows with different sequence lengths (should be 25, 50, 100)
    seq_len_data = metrics_df[metrics_df['Seq Length'].isin([25, 50, 100])].copy()
    
    if len(seq_len_data) == 0:
        print("Warning: No data with sequence lengths 25, 50, 100 found for plotting")
        return
    
    # Group by sequence length and get mean (in case of multiple runs)
    seq_len_data = seq_len_data.groupby('Seq Length').agg({
        'Accuracy': 'mean',
        'F1': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    x = seq_len_data['Seq Length']
    plt.plot(x, seq_len_data['Accuracy'], marker='o', label='Accuracy', linewidth=2)
    plt.plot(x, seq_len_data['F1'], marker='s', label='F1-score', linewidth=2)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Accuracy and F1-score vs Sequence Length', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_training_loss(history_best, history_worst, output_path='results/plots/training_loss_best_worst.png'):
    """Plot training loss for best and worst models."""
    plt.figure(figsize=(10, 6))
    
    if history_best:
        epochs_best = range(1, len(history_best['train_loss']) + 1)
        plt.plot(epochs_best, history_best['train_loss'], marker='o', 
                label=f"Best Model (Run {history_best.get('run_id', 'N/A')})", 
                linewidth=2, alpha=0.8)
    
    if history_worst:
        epochs_worst = range(1, len(history_worst['train_loss']) + 1)
        plt.plot(epochs_worst, history_worst['train_loss'], marker='s', 
                label=f"Worst Model (Run {history_worst.get('run_id', 'N/A')})", 
                linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss vs Epochs (Best and Worst Models)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


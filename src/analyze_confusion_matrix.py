"""
Generate confusion matrix and per-class metrics for the best model.
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed
from preprocess import load_and_preprocess_data
from models import get_model
from train import train_model


def evaluate_with_confusion_matrix(model, test_loader, device='cpu'):
    """Evaluate model and return predictions, labels, and probabilities."""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            probs = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, output_path='results/plots/confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix (Best Model)', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
    
    return cm


def analyze_best_model(data_path='data/IMDB Dataset.csv', 
                      vocab_cache_path='results/vocab_cache.pkl',
                      device='cpu', verbose=True):
    """Train best model and generate confusion matrix and per-class metrics."""
    
    # Best config: BiLSTM + Sigmoid + Adam + seq_len=100 + grad_clip=on
    config = {
        'arch': 'bilstm',
        'activation': 'sigmoid',
        'optimizer': 'adam',
        'seq_length': 100,
        'grad_clip': True
    }
    
    if verbose:
        print("Training best model for confusion matrix analysis...")
    
    # Set seed
    set_seed(42)
    
    # Load data
    train_loader, test_loader, vocab = load_and_preprocess_data(
        data_path=data_path,
        seq_length=config['seq_length'],
        batch_size=32,
        vocab_cache_path=vocab_cache_path
    )
    
    # Create model
    model = get_model(
        arch_type=config['arch'],
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2,
        dropout=0.4,
        activation=config['activation']
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer_name=config['optimizer'],
        learning_rate=0.001,
        max_epochs=10,
        patience=2,
        grad_clip=config['grad_clip'],
        device=device,
        verbose=verbose
    )
    
    # Evaluate with confusion matrix
    y_true, y_pred, y_probs = evaluate_with_confusion_matrix(model, test_loader, device=device)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Print results
    print("\n" + "="*70)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*70)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"True Negative    {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"True Positive    {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 60)
    print(f"{'Negative':<12} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<12}")
    print(f"{'Positive':<12} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<12}")
    
    # Compute error rates
    fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    print(f"\nError Rates:")
    print(f"  False Positive Rate (neg→pos): {fp_rate*100:.2f}%")
    print(f"  False Negative Rate (pos→neg): {fn_rate*100:.2f}%")
    print("="*70)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, 'results/plots/confusion_matrix.png')
    
    # Save per-class metrics
    metrics_df = pd.DataFrame({
        'Class': ['Negative', 'Positive'],
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Support': support
    })
    
    output_path = 'results/per_class_metrics.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"\nPer-class metrics saved to {output_path}")
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate
    }


if __name__ == '__main__':
    device = torch.device('cpu')
    print(f"Using device: {device}")
    analyze_best_model(device=device, verbose=True)


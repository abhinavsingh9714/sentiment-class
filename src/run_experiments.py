"""
Main experiment runner for RNN sentiment classification.
Runs 10 experiments: baseline + 5 factor sweeps.
"""
import argparse
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, log_hardware_info, save_metrics_to_csv, plot_accuracy_f1_vs_seqlen, plot_training_loss
from preprocess import load_and_preprocess_data
from models import get_model
from train import train_model
from evaluate import evaluate_model


def define_experiments():
    """
    Define the 10 experiments:
    - Baseline (1 run): BiLSTM + Tanh + Adam + seq_len=100 + grad_clip=on
    - Architecture sweep (2 runs): RNN, LSTM
    - Activation sweep (2 runs): ReLU, Sigmoid
    - Optimizer sweep (2 runs): SGD, RMSprop
    - Sequence length sweep (2 runs): 25, 50
    - Gradient clipping sweep (1 run): off
    """
    experiments = [
        # Baseline
        {
            'run_id': 'baseline',
            'arch': 'bilstm',
            'activation': 'tanh',
            'optimizer': 'adam',
            'seq_length': 100,
            'grad_clip': True
        },
        # Architecture sweep
        {
            'run_id': 'arch_rnn',
            'arch': 'rnn',
            'activation': 'tanh',
            'optimizer': 'adam',
            'seq_length': 100,
            'grad_clip': True
        },
        {
            'run_id': 'arch_lstm',
            'arch': 'lstm',
            'activation': 'tanh',
            'optimizer': 'adam',
            'seq_length': 100,
            'grad_clip': True
        },
        # Activation sweep
        {
            'run_id': 'act_relu',
            'arch': 'bilstm',
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': 100,
            'grad_clip': True
        },
        {
            'run_id': 'act_sigmoid',
            'arch': 'bilstm',
            'activation': 'sigmoid',
            'optimizer': 'adam',
            'seq_length': 100,
            'grad_clip': True
        },
        # Optimizer sweep
        {
            'run_id': 'opt_sgd',
            'arch': 'bilstm',
            'activation': 'tanh',
            'optimizer': 'sgd',
            'seq_length': 100,
            'grad_clip': True
        },
        {
            'run_id': 'opt_rmsprop',
            'arch': 'bilstm',
            'activation': 'tanh',
            'optimizer': 'rmsprop',
            'seq_length': 100,
            'grad_clip': True
        },
        # Sequence length sweep
        {
            'run_id': 'seq_25',
            'arch': 'bilstm',
            'activation': 'tanh',
            'optimizer': 'adam',
            'seq_length': 25,
            'grad_clip': True
        },
        {
            'run_id': 'seq_50',
            'arch': 'bilstm',
            'activation': 'tanh',
            'optimizer': 'adam',
            'seq_length': 50,
            'grad_clip': True
        },
        # Gradient clipping sweep
        {
            'run_id': 'no_clip',
            'arch': 'bilstm',
            'activation': 'tanh',
            'optimizer': 'adam',
            'seq_length': 100,
            'grad_clip': False
        }
    ]
    
    return experiments


def run_single_experiment(exp_config, data_path, vocab_cache_path, device='cpu', verbose=True):
    """
    Run a single experiment.
    
    Args:
        exp_config: Experiment configuration dictionary
        data_path: Path to dataset CSV
        vocab_cache_path: Path to vocabulary cache
        device: Device to run on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results
    """
    run_id = exp_config['run_id']
    
    if verbose:
        print("\n" + "="*70)
        print(f"Running Experiment: {run_id}")
        print(f"Architecture: {exp_config['arch']}, Activation: {exp_config['activation']}, "
              f"Optimizer: {exp_config['optimizer']}, Seq Length: {exp_config['seq_length']}, "
              f"Grad Clip: {exp_config['grad_clip']}")
        print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load and preprocess data
    train_loader, test_loader, vocab = load_and_preprocess_data(
        data_path=data_path,
        seq_length=exp_config['seq_length'],
        batch_size=32,
        vocab_cache_path=vocab_cache_path
    )
    
    # Create model
    model = get_model(
        arch_type=exp_config['arch'],
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2,
        dropout=0.4,
        activation=exp_config['activation']
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for simplicity
        optimizer_name=exp_config['optimizer'],
        learning_rate=0.001,
        max_epochs=10,
        patience=2,
        grad_clip=exp_config['grad_clip'],
        device=device,
        verbose=verbose
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device=device)
    
    # Prepare results
    results = {
        'Run ID': run_id,
        'Model': exp_config['arch'].upper(),
        'Activation': exp_config['activation'].capitalize(),
        'Optimizer': exp_config['optimizer'].upper(),
        'Seq Length': exp_config['seq_length'],
        'Grad Clipping': 'Yes' if exp_config['grad_clip'] else 'No',
        'Accuracy': round(metrics['accuracy'], 4),
        'F1': round(metrics['f1'], 4),
        'Epoch Time (s)': round(history['avg_epoch_time'], 2)
    }
    
    # Add history for plotting
    results['history'] = history
    results['run_id'] = run_id
    
    if verbose:
        print(f"\nResults for {run_id}:")
        print(f"  Accuracy: {results['Accuracy']:.4f}")
        print(f"  F1-score: {results['F1']:.4f}")
        print(f"  Avg Epoch Time: {results['Epoch Time (s)']:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run RNN sentiment classification experiments')
    parser.add_argument('--data_path', type=str, default='data/IMDB Dataset.csv',
                       help='Path to IMDB Dataset CSV')
    parser.add_argument('--vocab_cache', type=str, default='results/vocab_cache.pkl',
                       help='Path to vocabulary cache')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Log hardware info
    log_hardware_info()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Define experiments
    experiments = define_experiments()
    print(f"\nTotal experiments to run: {len(experiments)}")
    
    # Run all experiments
    all_results = []
    all_histories = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'='*70}")
        
        try:
            results = run_single_experiment(
                exp_config=exp_config,
                data_path=args.data_path,
                vocab_cache_path=args.vocab_cache,
                device=device,
                verbose=args.verbose
            )
            
            # Separate metrics from history
            history = results.pop('history')
            run_id = results.pop('run_id')
            
            all_results.append(results)
            all_histories.append({
                'run_id': run_id,
                'history': history,
                'results': results
            })
            
        except Exception as e:
            print(f"Error in experiment {exp_config['run_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save metrics to CSV
    if all_results:
        save_metrics_to_csv(all_results, 'results/metrics.csv')
        
        # Create plots
        import pandas as pd
        metrics_df = pd.DataFrame(all_results)
        
        # Plot accuracy/F1 vs sequence length
        plot_accuracy_f1_vs_seqlen(metrics_df, 'results/plots/accuracy_f1_vs_seqlen.png')
        
        # Find best and worst models based on F1-score
        best_idx = metrics_df['F1'].idxmax()
        worst_idx = metrics_df['F1'].idxmin()
        
        best_run_id = metrics_df.loc[best_idx, 'Run ID']
        worst_run_id = metrics_df.loc[worst_idx, 'Run ID']
        
        best_history = None
        worst_history = None
        
        for h in all_histories:
            if h['run_id'] == best_run_id:
                best_history = h['history']
                best_history['run_id'] = best_run_id
            if h['run_id'] == worst_run_id:
                worst_history = h['history']
                worst_history['run_id'] = worst_run_id
        
        # Plot training loss
        plot_training_loss(best_history, worst_history, 'results/plots/training_loss_best_worst.png')
        
        print("\n" + "="*70)
        print("All experiments completed!")
        print(f"Results saved to: results/metrics.csv")
        print(f"Plots saved to: results/plots/")
        print("="*70)
    else:
        print("No experiments completed successfully.")


if __name__ == '__main__':
    main()


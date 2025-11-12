"""
Run robustness experiments: best and worst configs with 3 different seeds.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, log_hardware_info
from preprocess import load_and_preprocess_data
from models import get_model
from train import train_model
from evaluate import evaluate_model
import pandas as pd
import numpy as np


def run_robustness_experiments(data_path='data/IMDB Dataset.csv', 
                               vocab_cache_path='results/vocab_cache.pkl',
                               device='cpu', verbose=True):
    """Run best and worst configs with 3 different seeds."""
    
    seeds = [41, 42, 43]
    
    # Best config: BiLSTM + Sigmoid + Adam + seq_len=100 + grad_clip=on
    best_config = {
        'arch': 'bilstm',
        'activation': 'sigmoid',
        'optimizer': 'adam',
        'seq_length': 100,
        'grad_clip': True
    }
    
    # Worst config: BiLSTM + Tanh + SGD + seq_len=100 + grad_clip=on
    worst_config = {
        'arch': 'bilstm',
        'activation': 'tanh',
        'optimizer': 'sgd',
        'seq_length': 100,
        'grad_clip': True
    }
    
    results = []
    
    for config_name, config in [('best', best_config), ('worst', worst_config)]:
        config_results = []
        
        for seed in seeds:
            run_id = f"{config_name}_seed{seed}"
            
            if verbose:
                print("\n" + "="*70)
                print(f"Running: {run_id}")
                print(f"Config: {config}")
                print("="*70)
            
            # Set seed
            set_seed(seed)
            
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
                val_loader=test_loader,  # Note: using test as val (same as original)
                optimizer_name=config['optimizer'],
                learning_rate=0.001,
                max_epochs=10,
                patience=2,
                grad_clip=config['grad_clip'],
                device=device,
                verbose=verbose
            )
            
            # Evaluate
            metrics = evaluate_model(model, test_loader, device=device)
            
            result = {
                'Config': config_name,
                'Seed': seed,
                'Accuracy': metrics['accuracy'],
                'F1': metrics['f1']
            }
            
            config_results.append(result)
            results.append(result)
            
            if verbose:
                print(f"Results: Acc={result['Accuracy']:.4f}, F1={result['F1']:.4f}")
        
        # Compute statistics for this config
        accs = [r['Accuracy'] for r in config_results]
        f1s = [r['F1'] for r in config_results]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        
        if verbose:
            print(f"\n{config_name.upper()} Config Statistics:")
            print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  F1-score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = 'results/robustness_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nRobustness results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    log_hardware_info()
    device = torch.device('cpu')
    print(f"Using device: {device}")
    run_robustness_experiments(device=device, verbose=True)


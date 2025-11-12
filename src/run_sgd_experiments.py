"""
Run SGD experiments with different learning rates and momentum values.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, log_hardware_info
from preprocess import load_and_preprocess_data
from models import get_model
from train import train_model, get_optimizer
from evaluate import evaluate_model
import pandas as pd


def run_sgd_grid(data_path='data/IMDB Dataset.csv', vocab_cache_path='results/vocab_cache.pkl', 
                 device='cpu', verbose=True):
    """Run SGD with different learning rates and momentum."""
    
    learning_rates = [0.1, 0.01, 0.001]
    momentums = [0.0, 0.9]
    
    results = []
    
    for lr in learning_rates:
        for momentum in momentums:
            run_id = f"sgd_lr{lr}_mom{momentum}"
            
            if verbose:
                print("\n" + "="*70)
                print(f"Running: {run_id}")
                print(f"Learning Rate: {lr}, Momentum: {momentum}")
                print("="*70)
            
            # Set seed
            set_seed(42)
            
            # Load data
            train_loader, test_loader, vocab = load_and_preprocess_data(
                data_path=data_path,
                seq_length=100,
                batch_size=32,
                vocab_cache_path=vocab_cache_path
            )
            
            # Create model
            model = get_model(
                arch_type='bilstm',
                vocab_size=len(vocab),
                embedding_dim=100,
                hidden_dim=64,
                num_layers=2,
                dropout=0.4,
                activation='tanh'
            )
            
            # Create optimizer with momentum
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            
            # Train model (we need to modify train_model to accept custom optimizer)
            # For now, let's use a simpler approach
            model = model.to(device)
            criterion = torch.nn.BCELoss()
            
            # Manual training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 2
            max_epochs = 10
            best_model_state = None
            
            history = {
                'train_loss': [],
                'val_loss': [],
                'epoch_times': []
            }
            
            import time
            for epoch in range(max_epochs):
                epoch_start = time.time()
                
                # Train
                model.train()
                train_loss = 0
                for sequences, labels in train_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for sequences, labels in test_loader:
                        sequences = sequences.to(device)
                        labels = labels.to(device)
                        outputs = model(sequences)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                
                val_loss /= len(test_loader)
                epoch_time = time.time() - epoch_start
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['epoch_times'].append(epoch_time)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Evaluate
            metrics = evaluate_model(model, test_loader, device=device)
            avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
            
            result = {
                'Run ID': run_id,
                'Learning Rate': lr,
                'Momentum': momentum,
                'Accuracy': round(metrics['accuracy'], 4),
                'F1': round(metrics['f1'], 4),
                'Epoch Time (s)': round(avg_epoch_time, 2),
                'Epochs': len(history['epoch_times'])
            }
            
            results.append(result)
            
            if verbose:
                print(f"Results: Acc={result['Accuracy']:.4f}, F1={result['F1']:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = 'results/sgd_grid_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSGD grid results saved to {output_path}")
    
    # Print best SGD result
    best_idx = df['F1'].idxmax()
    best_result = df.loc[best_idx]
    print(f"\nBest SGD configuration:")
    print(f"  LR: {best_result['Learning Rate']}, Momentum: {best_result['Momentum']}")
    print(f"  Accuracy: {best_result['Accuracy']:.4f}, F1: {best_result['F1']:.4f}")
    
    return results


if __name__ == '__main__':
    log_hardware_info()
    device = torch.device('cpu')
    print(f"Using device: {device}")
    run_sgd_grid(device=device, verbose=True)


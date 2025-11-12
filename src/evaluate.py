"""
Evaluation functions for RNN sentiment classification.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Dictionary with accuracy and F1-score (macro)
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            predictions = (outputs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1
    }
    
    return metrics


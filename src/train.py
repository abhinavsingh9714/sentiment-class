"""
Training functions for RNN sentiment classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time


def get_optimizer(model, optimizer_name, learning_rate=0.001):
    """
    Get optimizer based on name.
    
    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'sgd', or 'rmsprop'
        learning_rate: Learning rate (default: 0.001)
    
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, optimizer_name='adam', 
                learning_rate=0.001, max_epochs=10, patience=2, 
                grad_clip=False, device='cpu', verbose=True):
    """
    Train model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer_name: 'adam', 'sgd', or 'rmsprop'
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        grad_clip: Whether to use gradient clipping
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history and metrics
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, optimizer_name, learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, grad_clip)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        if verbose:
            print(f"Epoch {epoch+1}/{max_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Time: {epoch_time:.2f}s")
        
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
    
    # Calculate average epoch time
    avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
    history['avg_epoch_time'] = avg_epoch_time
    
    return history


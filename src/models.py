"""
RNN model architectures for sentiment classification.
"""
import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """Standard RNN model with configurable activation."""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_layers=2, 
                 dropout=0.4, activation='tanh'):
        super(RNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Activation function selection
        if activation == 'tanh':
            self.activation = nn.Tanh()
            rnn_nonlinearity = 'tanh'
            self.use_custom_activation = False
        elif activation == 'relu':
            self.activation = nn.ReLU()
            rnn_nonlinearity = 'relu'
            self.use_custom_activation = False
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            rnn_nonlinearity = 'tanh'  # RNN doesn't support sigmoid, use tanh then apply sigmoid
            self.use_custom_activation = True
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # RNN layers
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         nonlinearity=rnn_nonlinearity)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # RNN forward
        rnn_out, hidden = self.rnn(embedded)
        
        # Apply custom activation if needed
        if self.use_custom_activation:
            rnn_out = self.activation(rnn_out)
        
        # Take the last output
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)  # (batch_size, 1)
        
        # Sigmoid for binary classification
        output = self.sigmoid(output)
        
        return output.squeeze(1)  # (batch_size,)


class LSTMModel(nn.Module):
    """LSTM model for sentiment classification."""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_layers=2, 
                 dropout=0.4, activation='tanh'):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Activation function (applied to output if needed)
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.use_activation = True
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Apply activation
        if self.use_activation:
            last_output = self.activation(last_output)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)  # (batch_size, 1)
        
        # Sigmoid for binary classification
        output = self.sigmoid(output)
        
        return output.squeeze(1)  # (batch_size,)


class BidirectionalLSTMModel(nn.Module):
    """Bidirectional LSTM model for sentiment classification."""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_layers=2, 
                 dropout=0.4, activation='tanh'):
        super(BidirectionalLSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.use_activation = True
        self.dropout = nn.Dropout(dropout)
        # Bidirectional LSTM outputs 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Bidirectional LSTM forward
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Take the last output (concatenated forward and backward)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Apply activation
        if self.use_activation:
            last_output = self.activation(last_output)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)  # (batch_size, 1)
        
        # Sigmoid for binary classification
        output = self.sigmoid(output)
        
        return output.squeeze(1)  # (batch_size,)


def get_model(arch_type, vocab_size, embedding_dim=100, hidden_dim=64, 
              num_layers=2, dropout=0.4, activation='tanh'):
    """
    Factory function to get model based on architecture type.
    
    Args:
        arch_type: 'rnn', 'lstm', or 'bilstm'
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension (default: 100)
        hidden_dim: Hidden dimension (default: 64)
        num_layers: Number of layers (default: 2)
        dropout: Dropout rate (default: 0.4)
        activation: Activation function ('tanh', 'relu', or 'sigmoid')
    
    Returns:
        Model instance
    """
    if arch_type.lower() == 'rnn':
        return RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, activation)
    elif arch_type.lower() == 'lstm':
        return LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, activation)
    elif arch_type.lower() == 'bilstm':
        return BidirectionalLSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, activation)
    else:
        raise ValueError(f"Unsupported architecture: {arch_type}")


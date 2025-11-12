"""
Data preprocessing for IMDb sentiment classification.
"""
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os


class IMDBDataset(Dataset):
    """Dataset class for IMDb reviews."""
    
    def __init__(self, reviews, labels, vocab, seq_length):
        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        
        # Convert to sequence of token IDs
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in review]
        
        # Pad or truncate to fixed length
        if len(sequence) < self.seq_length:
            sequence = sequence + [self.vocab['<PAD>']] * (self.seq_length - len(sequence))
        else:
            sequence = sequence[:self.seq_length]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)


def clean_text(text):
    """Clean and preprocess text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Tokenize by splitting on whitespace
    tokens = text.split()
    return tokens


def build_vocab(reviews, vocab_size=10000):
    """Build vocabulary from reviews."""
    word_counts = Counter()
    
    # Count all words
    for review in reviews:
        word_counts.update(review)
    
    # Get top vocab_size most frequent words
    most_common = word_counts.most_common(vocab_size - 2)  # -2 for <PAD> and <UNK>
    
    # Create vocabulary dictionary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx
    
    return vocab


def load_and_preprocess_data(data_path, seq_length=100, batch_size=32, vocab_cache_path='results/vocab_cache.pkl'):
    """
    Load and preprocess IMDb dataset.
    
    Args:
        data_path: Path to IMDB Dataset.csv
        seq_length: Sequence length for padding/truncation
        batch_size: Batch size for DataLoader
        vocab_cache_path: Path to cache vocabulary
    
    Returns:
        train_loader, test_loader, vocab
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Split 50/50 (25k train, 25k test)
    train_df = df[:25000]
    test_df = df[25000:]
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Clean and tokenize reviews
    print("Preprocessing text...")
    train_reviews = [clean_text(review) for review in train_df['review'].values]
    test_reviews = [clean_text(review) for review in test_df['review'].values]
    
    # Build or load vocabulary
    if os.path.exists(vocab_cache_path):
        print(f"Loading cached vocabulary from {vocab_cache_path}...")
        with open(vocab_cache_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        print("Building vocabulary from top 10,000 words...")
        vocab = build_vocab(train_reviews, vocab_size=10000)
        os.makedirs(os.path.dirname(vocab_cache_path), exist_ok=True)
        with open(vocab_cache_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary cached to {vocab_cache_path}")
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert labels to binary (positive=1, negative=0)
    train_labels = (train_df['sentiment'] == 'positive').astype(int).values
    test_labels = (test_df['sentiment'] == 'positive').astype(int).values
    
    # Create datasets
    train_dataset = IMDBDataset(train_reviews, train_labels, vocab, seq_length)
    test_dataset = IMDBDataset(test_reviews, test_labels, vocab, seq_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaders created with sequence length: {seq_length}")
    
    return train_loader, test_loader, vocab


"""
Analyze dataset statistics for the report.
Computes token lengths, vocab coverage, truncation rates, etc.
"""
import pandas as pd
import numpy as np
from collections import Counter
from preprocess import clean_text, build_vocab
import pickle
import os


def analyze_dataset(data_path='data/IMDB Dataset.csv', vocab_cache_path='results/vocab_cache.pkl'):
    """Analyze dataset and compute statistics."""
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Split 50/50
    train_df = df[:25000]
    test_df = df[25000:]
    
    print("Preprocessing reviews...")
    train_reviews = [clean_text(review) for review in train_df['review'].values]
    test_reviews = [clean_text(review) for review in test_df['review'].values]
    
    # Compute token lengths
    train_lengths = [len(review) for review in train_reviews]
    test_lengths = [len(review) for review in test_reviews]
    all_lengths = train_lengths + test_lengths
    
    # Load or build vocabulary
    if os.path.exists(vocab_cache_path):
        with open(vocab_cache_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = build_vocab(train_reviews, vocab_size=10000)
    
    # Compute vocab coverage
    train_word_counts = Counter()
    for review in train_reviews:
        train_word_counts.update(review)
    
    test_word_counts = Counter()
    for review in test_reviews:
        test_word_counts.update(review)
    
    # Compute OOV rate on test set
    total_test_tokens = sum(test_word_counts.values())
    oov_test_tokens = sum(count for word, count in test_word_counts.items() if word not in vocab)
    oov_rate = oov_test_tokens / total_test_tokens if total_test_tokens > 0 else 0
    
    # Compute coverage for top-k vocab
    sorted_words = sorted(train_word_counts.items(), key=lambda x: x[1], reverse=True)
    top_5k_words = set([word for word, _ in sorted_words[:5000]])
    top_10k_words = set([word for word, _ in sorted_words[:10000]])
    
    top_5k_coverage = sum(count for word, count in train_word_counts.items() if word in top_5k_words) / sum(train_word_counts.values())
    top_10k_coverage = sum(count for word, count in train_word_counts.items() if word in top_10k_words) / sum(train_word_counts.values())
    
    # Compute truncation rates
    trunc_25 = sum(1 for length in all_lengths if length > 25) / len(all_lengths)
    trunc_50 = sum(1 for length in all_lengths if length > 50) / len(all_lengths)
    trunc_100 = sum(1 for length in all_lengths if length > 100) / len(all_lengths)
    
    stats = {
        'num_reviews': len(df),
        'num_train': len(train_df),
        'num_test': len(test_df),
        'mean_length': np.mean(all_lengths),
        'median_length': np.median(all_lengths),
        'p90_length': np.percentile(all_lengths, 90),
        'p95_length': np.percentile(all_lengths, 95),
        'max_length': np.max(all_lengths),
        'vocab_size': len(vocab),
        'oov_rate_test': oov_rate,
        'top_5k_coverage': top_5k_coverage,
        'top_10k_coverage': top_10k_coverage,
        'trunc_rate_25': trunc_25,
        'trunc_rate_50': trunc_50,
        'trunc_rate_100': trunc_100,
    }
    
    return stats


if __name__ == '__main__':
    stats = analyze_dataset()
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total Reviews: {stats['num_reviews']:,}")
    print(f"Train Reviews: {stats['num_train']:,}")
    print(f"Test Reviews: {stats['num_test']:,}")
    print(f"\nToken Length Statistics:")
    print(f"  Mean: {stats['mean_length']:.1f}")
    print(f"  Median: {stats['median_length']:.1f}")
    print(f"  90th percentile: {stats['p90_length']:.1f}")
    print(f"  95th percentile: {stats['p95_length']:.1f}")
    print(f"  Max: {stats['max_length']}")
    print(f"\nVocabulary:")
    print(f"  Size: {stats['vocab_size']:,}")
    print(f"  OOV rate on test: {stats['oov_rate_test']*100:.2f}%")
    print(f"  Top-5k coverage: {stats['top_5k_coverage']*100:.2f}%")
    print(f"  Top-10k coverage: {stats['top_10k_coverage']*100:.2f}%")
    print(f"\nTruncation Rates:")
    print(f"  % truncated at seq_len=25: {stats['trunc_rate_25']*100:.2f}%")
    print(f"  % truncated at seq_len=50: {stats['trunc_rate_50']*100:.2f}%")
    print(f"  % truncated at seq_len=100: {stats['trunc_rate_100']*100:.2f}%")
    print("="*70)


# RNN Sentiment Classification

This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures for sentiment classification on the IMDb Movie Review Dataset, treating it as a sequence classification problem.

## Project Structure

```
sentiment-class/
├── data/
│   └── IMDB Dataset.csv          # Dataset (50,000 reviews)
├── src/
│   ├── preprocess.py             # Data preprocessing and loading
│   ├── models.py                 # RNN, LSTM, and Bidirectional LSTM models
│   ├── train.py                  # Training functions with early stopping
│   ├── evaluate.py               # Evaluation functions
│   ├── utils.py                  # Utility functions (seeds, plotting, etc.)
│   ├── run_experiments.py        # Main experiment runner
│   ├── analyze_dataset.py        # Dataset statistics analysis
│   ├── run_sgd_experiments.py    # SGD hyperparameter grid search
│   ├── run_robustness.py         # Robustness testing with multiple seeds
│   └── analyze_confusion_matrix.py # Confusion matrix and per-class metrics
├── results/
│   ├── metrics.csv               # Results table (generated)
│   ├── sgd_grid_results.csv      # SGD tuning results (generated)
│   ├── robustness_results.csv    # Robustness test results (generated)
│   ├── per_class_metrics.csv     # Per-class metrics for best model (generated)
│   ├── vocab_cache.pkl           # Cached vocabulary (generated)
│   └── plots/                    # Generated plots
│       ├── accuracy_f1_vs_seqlen.png
│       ├── training_loss_best_worst.png
│       └── confusion_matrix.png
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

### Python Version
- Python 3.8 or higher is required

### Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `torch` (>=2.0.0) - PyTorch for deep learning
- `numpy` (>=1.24.0) - Numerical computing
- `pandas` (>=2.0.0) - Data manipulation
- `matplotlib` (>=3.7.0) - Plotting
- `seaborn` (>=0.12.0) - Statistical visualization
- `scikit-learn` (>=1.3.0) - Metrics calculation
- `tqdm` (>=4.65.0) - Progress bars
- `psutil` (>=5.9.0) - System information

### Dataset

The IMDb Dataset CSV file should be placed in the `data/` directory. The dataset contains 50,000 movie reviews with binary sentiment labels (positive/negative).

## How to Run

### Running All Experiments

To run all 10 experiments systematically:

```bash
python src/run_experiments.py
```

This will:
1. Load and preprocess the dataset
2. Run 10 experiments (baseline + 5 factor sweeps)
3. Generate `results/metrics.csv` with all results
4. Generate plots in `results/plots/`

### Command Line Options

```bash
python src/run_experiments.py [OPTIONS]
```

Options:
- `--data_path`: Path to IMDB Dataset CSV (default: `data/IMDB Dataset.csv`)
- `--vocab_cache`: Path to vocabulary cache (default: `results/vocab_cache.pkl`)
- `--device`: Device to use - `cpu` or `cuda` (default: `cpu`)
- `--verbose`: Print detailed progress (default: True)

Example with GPU:
```bash
python src/run_experiments.py --device cuda
```

### Running Additional Analyses

After running the main experiments, you can run additional analyses:

#### 1. Dataset Statistics

Analyze dataset characteristics (token lengths, vocabulary coverage, truncation rates):

```bash
python src/analyze_dataset.py
```

This generates statistics about:
- Token length distribution (mean, median, percentiles)
- Vocabulary coverage (OOV rate, top-k coverage)
- Truncation rates at different sequence lengths

#### 2. SGD Hyperparameter Tuning

Run SGD with different learning rates and momentum values:

```bash
python src/run_sgd_experiments.py
```

This runs a grid search over:
- Learning rates: [0.1, 0.01, 0.001]
- Momentum: [0.0, 0.9]

Results are saved to `results/sgd_grid_results.csv`.

#### 3. Robustness Testing

Test model stability across different random seeds:

```bash
python src/run_robustness.py
```

This re-runs the best and worst configurations with seeds {41, 42, 43} to assess variance. Results are saved to `results/robustness_results.csv`.

#### 4. Confusion Matrix Analysis

Generate confusion matrix and per-class metrics for the best model:

```bash
python src/analyze_confusion_matrix.py
```

This:
- Trains the best model (BiLSTM + Sigmoid + Adam + seq_len=100)
- Generates confusion matrix plot (`results/plots/confusion_matrix.png`)
- Computes per-class precision, recall, and F1-score
- Saves metrics to `results/per_class_metrics.csv`

## Experimental Design

The project uses an efficient 10-run experimental design:

1. **Baseline (1 run)**: BiLSTM + Tanh + Adam + seq_len=100 + grad_clip=on
2. **Architecture sweep (2 runs)**: RNN, LSTM (holding other baseline parameters)
3. **Activation sweep (2 runs)**: ReLU, Sigmoid (holding other baseline parameters)
4. **Optimizer sweep (2 runs)**: SGD, RMSprop (holding other baseline parameters)
5. **Sequence length sweep (2 runs)**: 25, 50 (holding other baseline parameters)
6. **Gradient clipping sweep (1 run)**: off (holding other baseline parameters)

### Model Configuration

All models use:
- Embedding dimension: 100
- Hidden dimension: 64
- Number of layers: 2
- Dropout: 0.4 (fixed)
- Batch size: 32
- Learning rate: 0.001
- Early stopping: patience=2, max_epochs=10
- Loss function: Binary Cross-Entropy

## Expected Runtime

- **Per experiment**: Approximately 5-15 minutes (depending on hardware)
- **Total runtime**: Approximately 1-2.5 hours for all 10 experiments (CPU)
- **With GPU**: Significantly faster (approximately 3-5x speedup)
- **SGD grid search**: ~1-2 hours (6 configurations)
- **Robustness testing**: ~30-45 minutes (6 runs: 2 configs × 3 seeds)
- **Confusion matrix analysis**: ~10-15 minutes (trains best model once)

The actual runtime depends on:
- Hardware (CPU/GPU, number of cores)
- Sequence length (longer sequences take more time)
- Model architecture (BiLSTM is slower than RNN/LSTM)

## Output Files

### `results/metrics.csv`

Contains a table with all experiment results:
- Run ID
- Model (RNN/LSTM/BILSTM)
- Activation (Tanh/ReLU/Sigmoid)
- Optimizer (Adam/SGD/RMSprop)
- Seq Length (25/50/100)
- Grad Clipping (Yes/No)
- Accuracy
- F1-score (macro)
- Epoch Time (s)

### `results/plots/accuracy_f1_vs_seqlen.png`

Plot showing how Accuracy and F1-score vary with sequence length (using runs with seq_len 25, 50, 100).

### `results/plots/training_loss_best_worst.png`

Plot comparing training loss curves for the best and worst performing models (based on F1-score).

### `results/plots/confusion_matrix.png`

Confusion matrix visualization for the best model (BiLSTM + Sigmoid + Adam + seq_len=100). Generated by `analyze_confusion_matrix.py`.

### `results/sgd_grid_results.csv`

Results from SGD hyperparameter tuning grid search. Contains accuracy and F1-score for each (learning rate, momentum) combination. Generated by `run_sgd_experiments.py`.

### `results/robustness_results.csv`

Results from robustness testing across multiple seeds. Contains accuracy and F1-score for best and worst configurations with seeds {41, 42, 43}. Generated by `run_robustness.py`.

### `results/per_class_metrics.csv`

Per-class precision, recall, F1-score, and support for the best model. Generated by `analyze_confusion_matrix.py`.

### `results/vocab_cache.pkl`

Cached vocabulary built from the top 10,000 most frequent words. This is reused across experiments to speed up preprocessing.

## Hardware Requirements

- **Minimum RAM**: 8 GB
- **Recommended RAM**: 16 GB or more
- **CPU**: Multi-core processor recommended
- **GPU**: Optional but recommended for faster training (CUDA-compatible)

The code will automatically detect and use GPU if available. Hardware information is logged at the start of each run.

## Reproducibility

The code uses fixed random seeds (42) for:
- PyTorch
- NumPy
- Python's random module

Nondeterministic operations are disabled to ensure reproducibility. However, note that:
- GPU operations may still have some non-determinism
- Results may vary slightly between different hardware/software versions

## Key Features

- **Efficient preprocessing**: Vocabulary is cached and reused across experiments
- **Early stopping**: Prevents overfitting and reduces training time
- **Gradient clipping**: Optional stability strategy
- **Comprehensive evaluation**: Accuracy and F1-score (macro) metrics
- **Automatic plotting**: Generates required plots automatically
- **Hardware logging**: Records system information for reproducibility
- **Extended analysis**: Dataset statistics, SGD tuning, robustness testing, and confusion matrix analysis
- **Reproducibility**: Fixed random seeds and deterministic operations

## Troubleshooting

### Out of Memory Errors

If you encounter memory issues:
- Reduce batch size in `preprocess.py` (default: 32)
- Use smaller sequence lengths
- Process experiments one at a time

### Slow Training

- Use GPU if available: `--device cuda`
- Reduce sequence length
- Reduce number of epochs (modify `max_epochs` in `train.py`)

### Import Errors

Make sure you're running from the project root directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```

## License

This project is for educational purposes as part of a course assignment.


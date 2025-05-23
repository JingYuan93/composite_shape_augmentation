#!/usr/bin/env python3
"""
Command-line tool for Bayesian time-series step prediction with custom cross-validation cases.
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import itertools


class TimeSeriesDataset(Dataset):
    """Dataset wrapping sliding-window sequences for next-step prediction."""
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences) - 1

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx + 1]


# ----- Utility functions -----

def normalize_2d_array(arr: np.ndarray):
    col_max = arr.max(axis=0)
    col_min = arr.min(axis=0)
    normalized = (arr - col_min) / (col_max - col_min)
    return normalized, col_min, col_max


def read_normalize_and_split_csv(csv_file: str, split_size: int, num_splits: int, nrows: int):
    df = pd.read_csv(csv_file, header=None, nrows=nrows)
    # Insert delta in last column
    delta = [0, 0.5, 1, 2.5, 5, 7.5, 10, 17.5, 50, 100]
    if len(df) % len(delta) != 0:
        raise ValueError("CSV row count must be multiple of delta length.")
    rep = len(df) // len(delta)
    df.iloc[:, -1] = np.tile(delta, rep)
    # Split into equal parts
    splits = [df.iloc[i * split_size:(i + 1) * split_size].values for i in range(num_splits)]
    return splits


def sliding_windows(data: np.ndarray, window: int, step: int) -> np.ndarray:
    n_samples = (len(data) - window) // step + 1
    return np.stack([data[i * step : i * step + window] for i in range(n_samples)], axis=0)


# ----- Model definitions -----

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_sigma = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_sigma = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
        b = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        return F.linear(x, w, b)


class BayesianNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layer1 = BayesianLinear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


# ----- Training and prediction -----

def train_model(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, epochs: int):
    model.train()
    for _ in range(epochs):
        for seq in loader:
            x = seq[:, 0, :].float()
            y = seq[:, 1, :].float()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()


def generate_predictions(model: nn.Module, val_sequence: np.ndarray,
                         window_size: int, step_size: int, n_samples: int) -> np.ndarray:
    # Normalize and prepare DataLoader
    val_norm, MIN, MAX = normalize_2d_array(val_sequence)
    # Autoregressive eval_steps = len(val_norm)-1
    eval_steps = len(val_norm) - 1
    loader = DataLoader(TimeSeriesDataset(torch.tensor(val_norm)), batch_size=1)

    model.eval()
    preds = []
    # Denorm tensors
    min_t = torch.tensor(MIN, dtype=torch.float32).unsqueeze(0)
    max_t = torch.tensor(MAX, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prev = None
        for i, (seq, next_seq) in enumerate(loader):
            if i >= eval_steps:
                break
            x = seq.float() if i == 0 else prev
            # Monte Carlo sampling
            samples = torch.stack([model(x) for _ in range(n_samples)])
            mean_pred = samples.mean(0)
            # Denormalize
            denorm = mean_pred * (max_t - min_t) + min_t
            preds.append(denorm.squeeze(0).numpy())
            prev = mean_pred
    return np.array(preds)


def cross_validate(args):
    splits = read_normalize_and_split_csv(
        args.csv_file, args.split_size, args.num_splits, args.nrows)
    # Determine validation indices per case
    case_map = {
        'case1': [2, 3],
        'case2': [0, 3],
        'case3': [1, 3]
    }
    val_indices = case_map[args.case]
    train_indices = [i for i in range(args.num_splits) if i not in val_indices]

    # Prepare training data
    train_arr = np.concatenate([splits[i] for i in train_indices], axis=0)
    train_norm, _, _ = normalize_2d_array(train_arr)

    # DataLoader for training
    train_windows = sliding_windows(train_norm, args.window_size, args.step_size)
    train_loader = DataLoader(torch.tensor(train_windows), batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = BayesianNetwork(
        input_dim=train_windows.shape[2],
        hidden_dim=args.hidden_size,
        output_dim=train_windows.shape[2],
        dropout_rate=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train once
    train_model(model, train_loader, nn.L1Loss(), optimizer, args.epochs)

    # Generate predictions for each validation split
    predictions = {}
    for idx in val_indices:
        preds = generate_predictions(
            model,
            splits[idx],
            args.window_size,
            args.step_size,
            args.n_samples
        )
        predictions[f"split{idx+1}"] = preds

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian predictor with custom case selection for validation sets")
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--split_size", type=int, default=10)
    parser.add_argument("--num_splits", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case", type=str, choices=["case1","case2","case3"], required=True,
                        help="Select which case to use for validation splits")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    predictions = cross_validate(args)
    print(predictions)

if __name__ == "__main__":
    main()

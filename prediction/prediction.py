#!/usr/bin/env python3
"""
Command-line tool for Bayesian time-series step prediction with custom cross-validation cases,
now with publication-quality training & validation loss plotting and CSV export.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    """Dataset wrapping sliding-window sequences for next-step prediction."""
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences) - 1

    def __getitem__(self, idx):
        # Return full vector at t and t+1
        return self.sequences[idx], self.sequences[idx + 1]

# Utility functions

def normalize_2d_array(arr: np.ndarray):
    col_max = arr.max(axis=0)
    col_min = arr.min(axis=0)
    normalized = (arr - col_min) / (col_max - col_min)
    return normalized, col_min, col_max


def read_normalize_and_split_csv(csv_file: str, split_size: int, num_splits: int, nrows: int):
    df = pd.read_csv(csv_file, header=None, nrows=nrows)
    delta = [0, 0.5, 1, 2.5, 5, 7.5, 10, 17.5, 50, 100]
    rep = len(df) // len(delta)
    df.iloc[:, -1] = np.tile(delta, rep)
    splits = [df.iloc[i * split_size:(i + 1) * split_size].values for i in range(num_splits)]
    return splits


def sliding_windows(data: np.ndarray, window: int, step: int) -> np.ndarray:
    n_samples = (len(data) - window) // step + 1
    return np.stack([data[i * step: i * step + window] for i in range(n_samples)], axis=0)

# Model definitions

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

# Training & Validation

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for x_full, y_full in loader:
        # x_full, y_full: (batch, 37)
        x = x_full.float()
        y = y_full.float()[:, :-1]  
        optimizer.zero_grad()
        pred = model(x)            
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_full, y_full in loader:
            x = x_full.float()
            y = y_full.float()[:, :-1]
            total_loss += criterion(model(x), y).item()
    return total_loss / len(loader)


def generate_predictions(model: nn.Module, val_sequence: np.ndarray, window_size: int, step_size: int, n_samples: int) -> np.ndarray:
    val_norm_full, MIN_full, MAX_full = normalize_2d_array(val_sequence)
    MIN_feat = torch.tensor(MIN_full[:-1], dtype=torch.float32).unsqueeze(0)
    MAX_feat = torch.tensor(MAX_full[:-1], dtype=torch.float32).unsqueeze(0)
    delta_norm = val_norm_full[:, -1]

    loader = DataLoader(TimeSeriesDataset(torch.tensor(val_norm_full)), batch_size=1)
    model.eval()
    preds, prev = [], None
    with torch.no_grad():
        for i, (seq, _) in enumerate(loader):
            if i >= len(val_norm_full) - 1:
                break
            if i == 0:
                x = seq.float()
            else:
                
                delta_i = torch.tensor([[delta_norm[i]]], dtype=torch.float32)
                x = torch.cat([prev, delta_i], dim=1)
            samples = torch.stack([model(x) for _ in range(n_samples)])
            mean_pred = samples.mean(0)
            
            denorm = mean_pred * (MAX_feat - MIN_feat) + MIN_feat
            preds.append(denorm.squeeze(0).numpy())
            prev = mean_pred
    return np.array(preds)


def cross_validate(args):
    splits = read_normalize_and_split_csv(args.csv_file, args.split_size, args.num_splits, args.nrows)
    case_map = {'case1': [2,3], 'case2': [0,3], 'case3': [1,3]}
    val_indices = case_map[args.case]
    train_indices = [i for i in range(args.num_splits) if i not in val_indices]

    
    train_arr = np.concatenate([splits[i] for i in train_indices], axis=0)
    train_norm, _, _ = normalize_2d_array(train_arr)
    train_dataset = TimeSeriesDataset(torch.tensor(train_norm))
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    val_arr = np.concatenate([splits[i] for i in val_indices], axis=0)
    val_norm, _, _ = normalize_2d_array(val_arr)
    val_dataset = TimeSeriesDataset(torch.tensor(val_norm))
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    
    input_dim  = train_norm.shape[1]          
    output_dim = input_dim - 1                
    model = BayesianNetwork(input_dim=input_dim,
                             hidden_dim=args.hidden_size,
                             output_dim=output_dim,
                             dropout_rate=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss = validate(model, val_loader, criterion)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        print(f"Epoch {epoch}/{args.epochs} — train_loss: {tr_loss:.4f}, val_loss: {vl_loss:.4f}")
        
    results = {}
    for idx in val_indices:
        results[f"split{idx+1}"] = generate_predictions(
            model, splits[idx], args.window_size, args.step_size, args.n_samples)
    return results


def main():
    parser = argparse.ArgumentParser(description="Bayesian predictor with publication-quality plotting and CSV export")
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
    parser.add_argument("--case", choices=["case1","case2","case3"], required=True,
                        help="Select validation case")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    predictions = cross_validate(args)
    print(predictions)

if __name__ == "__main__":
    main()

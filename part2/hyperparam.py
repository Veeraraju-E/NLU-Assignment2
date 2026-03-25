"""
Standalone hyperparameter sweep for character-level name generation models.
"""

import argparse
import csv
import itertools
from pathlib import Path

import torch
from tqdm import tqdm

from dataloader import get_train_val_test_loaders
from model import VanillaRNN, BLSTM, RNNWithAttention
from train import eval_loss, train_epoch


def parse_int_list(value: str):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value: str):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def build_model(model_name: str, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int, dropout: float):
    if model_name == "vanilla_rnn":
        return VanillaRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "blstm":
        return BLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "rnn_attention":
        return RNNWithAttention(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for char-level name generation")
    parser.add_argument("--model", choices=["vanilla_rnn", "blstm", "rnn_attention"], default="vanilla_rnn")
    parser.add_argument("--data", default="part2/TrainingNames.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--embed_range", type=parse_int_list, default=[128], help="Comma-separated ints")
    parser.add_argument("--hidden_range", type=parse_int_list, default=[128], help="Comma-separated ints")
    parser.add_argument("--layers_range", type=parse_int_list, default=[2, 5, 8], help="Comma-separated ints")
    parser.add_argument("--dropout_range", type=parse_float_list, default=[0.1, 0.2, 0.3], help="Comma-separated floats")
    parser.add_argument("--lr_range", type=parse_float_list, default=[0.0001, 0.001], help="Comma-separated floats")
    parser.add_argument("--batch_range", type=parse_int_list, default=[128], help="Comma-separated ints")
    parser.add_argument("--max_trials", type=int, default=0, help="0 means run full grid")
    parser.add_argument("--results_csv", default="part2/hyperparam_sweep_results.csv", help="Single CSV file to store sweep results")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent.parent

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    results_csv_path = Path(args.results_csv)
    if not results_csv_path.is_absolute():
        results_csv_path = project_root / results_csv_path

    all_configs = list(
        itertools.product(
            args.embed_range,
            args.hidden_range,
            args.layers_range,
            args.dropout_range,
            args.lr_range,
            args.batch_range,
        )
    )
    if args.max_trials > 0:
        all_configs = all_configs[: args.max_trials]

    if len(all_configs) == 0:
        raise ValueError("No trials to run. Check your *_range arguments.")

    best = None
    total = len(all_configs)
    trial_rows = []
    trial_pbar = tqdm(all_configs, desc="Trials", total=total, dynamic_ncols=True)
    for idx, (embed_dim, hidden_size, num_layers, dropout, lr, batch_size) in enumerate(trial_pbar, start=1):
        train_loader, val_loader, _, char_to_idx, _ = get_train_val_test_loaders(
            str(data_path),
            batch_size=batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        model = build_model(args.model, len(char_to_idx), embed_dim, hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float("inf")
        best_val_epoch = -1

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, device, val_loader=None)
            epoch_val_loss = eval_loss(model, val_loader, device)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_epoch = epoch

        config = {
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
        }
        trial_pbar.set_postfix(best_val=f"{best_val_loss:.4f}", best_epoch=best_val_epoch)
        print(f"[{idx}/{total}] best_val_loss={best_val_loss:.4f} best_val_epoch={best_val_epoch} config={config}")
        trial_rows.append(
            {
                "trial_index": idx,
                "model": args.model,
                "embed_dim": embed_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
                "best_val_loss": best_val_loss,
                "best_val_epoch": best_val_epoch,
            }
        )

        if best is None or best_val_loss < best["val_loss"]:
            best = {"val_loss": best_val_loss, "best_epoch": best_val_epoch, "config": config}

    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_index",
        "model",
        "embed_dim",
        "hidden_size",
        "num_layers",
        "dropout",
        "lr",
        "batch_size",
        "best_val_loss",
        "best_val_epoch",
    ]
    with results_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)

    print("\nBest trial:")
    print(f"val_loss={best['val_loss']:.4f}")
    print(f"best_epoch={best['best_epoch']}")
    print(f"config={best['config']}")
    print(f"results_csv={results_csv_path}")


if __name__ == "__main__":
    main()

# imports
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from dataloader import get_train_val_test_loaders
from model import VanillaRNN, BLSTM, RNNWithAttention
from train import generate, train_epoch

# utilities/helpers
def get_checkpoint_paths(model_name: str, base_dir: Path):
    ckpt_dir = base_dir / "checkpoints"
    return ckpt_dir / f"{model_name}_best.pt", ckpt_dir / f"{model_name}_latest.pt"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    char_to_idx: dict,
    idx_to_char: dict,
    best_val_loss: float = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }
    if best_val_loss is not None:
        d["best_val_loss"] = best_val_loss
    torch.save(d, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return (
        ckpt.get("epoch", -1) + 1,
        ckpt.get("char_to_idx"),
        ckpt.get("idx_to_char"),
        ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))),
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # the best hyperparameters after grid search were:
    # model: vanilla_rnn
    # embed_dim: 256
    # hidden_size: 256
    # num_layers: 2
    # dropout: 0.1
    # lr: 0.001
    # batch_size: 128

    # model: rnn_attention
    # embed_dim: 128
    # hidden_size: 256
    # num_layers: 2
    # dropout: 0.2
    # lr: 0.001
    # batch_size: 128

    parser = argparse.ArgumentParser(description="Character-level name generation (Task 1)")
    parser.add_argument("--model", choices=["vanilla_rnn", "blstm", "rnn_attention"], default="vanilla_rnn", help="Model variant to train")
    parser.add_argument("--data", default="part2/TrainingNames.txt", help="Path to TrainingNames.txt")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--embed", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of RNN/LSTM layers")
    parser.add_argument("--samples", type=int, default=20, help="Number of generated samples to print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="part2", help="Base dir for checkpoints")
    parser.add_argument("--no_resume", action="store_true", help="Skip loading checkpoint even if exists")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent.parent
    ckpt_best_path, ckpt_latest_path = get_checkpoint_paths(args.model, base_dir / args.checkpoint_dir)

    # Resolve data path relative to project root
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    train_loader, val_loader, test_loader, char_to_idx, idx_to_char = get_train_val_test_loaders(str(data_path), batch_size=args.batch_size, train_ratio=0.7, val_ratio=0.2, seed=args.seed)
    vocab_size = len(char_to_idx)

    if args.model == "vanilla_rnn":
        model = VanillaRNN(vocab_size=vocab_size, embed_dim=args.embed, hidden_size=args.hidden, num_layers=args.layers)
    elif args.model == "blstm":
        model = BLSTM(vocab_size=vocab_size, embed_dim=args.embed, hidden_size=args.hidden, num_layers=args.layers)
    else:
        model = RNNWithAttention(vocab_size=vocab_size, embed_dim=args.embed, hidden_size=args.hidden, num_layers=args.layers)

    n_params = count_trainable_parameters(model)
    print(f"Model ({args.model}) trainable parameters: {n_params}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_val_loss = float("inf")

    if ckpt_latest_path.exists() and not args.no_resume:
        start_epoch, ckpt_cti, ckpt_itc, best_val_loss = load_checkpoint(ckpt_latest_path, model, optimizer, device)
        if ckpt_cti is not None:
            char_to_idx = ckpt_cti
        if ckpt_itc is not None:
            idx_to_char = {int(k): v for k, v in ckpt_itc.items()}

    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Epoch", position=0, leave=True, dynamic_ncols=True)
    for ep in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {ep + 1}/{args.epochs}")
        train_loss, val_loss = train_epoch(model, train_loader, optimizer, device, val_loader=val_loader, epoch_pbar=epoch_pbar)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_best_path, model, optimizer, ep, train_loss, val_loss, char_to_idx, idx_to_char)
        save_checkpoint(ckpt_latest_path, model, optimizer, ep, train_loss, val_loss, char_to_idx, idx_to_char, best_val_loss=best_val_loss)

    samples = generate(model, idx_to_char, char_to_idx, device, n_samples=args.samples)
    print("\nGenerated samples:")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    main()

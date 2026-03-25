# main training script
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import PAD_IDX


def eval_loss(model: nn.Module, loader, device: torch.device) -> float:
    """Compute average loss on loader without backward."""
    model.eval()
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    with torch.no_grad():
        for inp, tgt, lengths in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            lengths = lengths.to(device)
            logits = model(inp, lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += loss.item() * inp.size(0)
            n += inp.size(0)
    return total_loss / n if n > 0 else 0.0


def train_epoch(model: nn.Module, train_loader, optimizer: torch.optim.Optimizer, device: torch.device, val_loader=None, epoch_pbar=None) -> Tuple[float, float]:
    """One epoch of training. Returns (train_loss, val_loss). If epoch_pbar given, updates it with both."""
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    batch_pbar = tqdm(train_loader, leave=False, position=1)
    for inp, tgt, lengths in batch_pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad()
        logits = model(inp, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * inp.size(0)
        n += inp.size(0)
        avg_train = total_loss / n
        batch_pbar.set_postfix(train=f"{avg_train:.4f}")
    train_loss = total_loss / n if n > 0 else 0.0
    val_loss = eval_loss(model, val_loader, device) if val_loader is not None else 0.0
    if epoch_pbar is not None:
        epoch_pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
        epoch_pbar.refresh()
    return train_loss, val_loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate(model: nn.Module, idx_to_char: Dict[int, str], char_to_idx: Dict[str, int], device: torch.device, n_samples: int = 10, max_len: int = 30, temperature: float = 0.8) -> List[str]:
    """Generate names by sampling. Starts from random char, stops at EOS or max_len."""
    model.eval()
    EOS_IDX = char_to_idx.get("\n", 1)
    pad_idx = char_to_idx.get("\0", 0)
    valid_start = [i for i in idx_to_char if i != pad_idx and i != EOS_IDX]
    names = []
    with torch.no_grad():
        for _ in range(n_samples):
            start_idx = valid_start[torch.randint(0, len(valid_start), (1,)).item()]
            seq = [start_idx]
            for _ in range(max_len - 1):
                x = torch.tensor([seq], dtype=torch.long, device=device)
                lengths = torch.tensor([len(seq)], dtype=torch.long, device=device)
                logits = model(x, lengths)
                logits = logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                if next_idx == EOS_IDX:
                    break
                seq.append(next_idx)
            s = "".join(idx_to_char[i] for i in seq if i != pad_idx)
            names.append(s)
    return names

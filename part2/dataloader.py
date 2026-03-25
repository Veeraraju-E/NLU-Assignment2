from pathlib import Path
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader


# Special token
PAD_CHAR = "\0"
PAD_IDX = 0
EOS_CHAR = "\n"


def load_names(path: str) -> List[str]:
    """Load names from file, one per line."""
    p = Path(path)
    names = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            names.append(line + EOS_CHAR)
    return names


def build_vocab(names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build char->idx and idx->char mappings. PAD=0, other chars by frequency."""
    char_counts: Dict[str, int] = {}
    for name in names:
        for c in name:
            char_counts[c] = char_counts.get(c, 0) + 1
    chars_sorted = sorted(char_counts.keys(), key=lambda x: (-char_counts[x], x))
    char_to_idx = {PAD_CHAR: PAD_IDX}
    for i, c in enumerate(chars_sorted, start=1):
        char_to_idx[c] = i
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode_sequence(seq: str, char_to_idx: Dict[str, int]) -> List[int]:
    return [char_to_idx[c] for c in seq]


class NameDataset(Dataset):
    """Dataset of names as character sequences. Each sample: (input_ids, target_ids)."""

    def __init__(self, names: List[str], char_to_idx: Dict[str, int]):
        self.names = names
        self.char_to_idx = char_to_idx
        self.samples = []
        for name in names:
            inp = encode_sequence(name[:-1], char_to_idx)
            tgt = encode_sequence(name[1:], char_to_idx)
            self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


def collate_pad(batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to max length in batch. Returns (input, target, lengths)."""
    max_len = max(len(inp) for inp, _ in batch)
    inp_padded = []
    tgt_padded = []
    lengths = []
    for inp, tgt in batch:
        lengths.append(len(inp))
        inp_padded.append(inp + [PAD_IDX] * (max_len - len(inp)))
        tgt_padded.append(tgt + [PAD_IDX] * (max_len - len(tgt)))
    return (
        torch.tensor(inp_padded, dtype=torch.long),
        torch.tensor(tgt_padded, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def get_dataloader(data_path: str, batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, Dict[str, int], Dict[int, str]]:
    """Load names and build vocab, return DataLoader and vocab dicts."""
    names = load_names(data_path)
    char_to_idx, idx_to_char = build_vocab(names)
    dataset = NameDataset(names, char_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pad)
    return loader, char_to_idx, idx_to_char


def get_train_val_test_loaders(data_path: str, batch_size: int = 32, train_ratio: float = 0.7, val_ratio: float = 0.2, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Split data into train/val/test loaders and build vocab."""
    names = load_names(data_path)
    random.seed(seed)
    random.shuffle(names)
    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_names = names[:n_train]
    val_names = names[n_train : n_train + n_val]
    test_names = names[n_train + n_val :]
    char_to_idx, idx_to_char = build_vocab(names)
    train_ds = NameDataset(train_names, char_to_idx)
    val_ds = NameDataset(val_names, char_to_idx)
    test_ds = NameDataset(test_names, char_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    return train_loader, val_loader, test_loader, char_to_idx, idx_to_char

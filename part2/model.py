# model definitions

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaRNN(nn.Module):
    """
    Vanilla RNN: embedding -> stacked RNN -> linear -> logits.
    Architecture: char_embedding -> RNN (tanh) -> hidden -> Linear -> vocab_logits.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T), lengths: (B,)
        emb = self.embed(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed, h)
        out, _ = pad_packed_sequence(out, batch_first=True)
        logits = self.fc(out)
        return logits


class BLSTM(nn.Module):
    """
    Bidirectional LSTM: embedding -> BLSTM -> concat fwd/bwd -> linear -> logits.
    At each step t, forward sees 0..t, backward sees T..t; concat used for prediction.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        logits = self.fc(out)
        return logits


class RNNWithAttention(nn.Module):
    """
    RNN with basic additive attention over encoder hidden states.
    Encoder RNN produces h_1..h_T. For step t, attend over h_1..h_t (causal),
    get context, combine with h_t, project to vocab.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity="tanh",
        )
        self.attn_W = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # out: (B, T, H). Causal attention at step t over h_0..h_t
        B, T, H = out.shape
        device = out.device
        logits_list = []
        for t in range(T):
            h_t = out[:, t, :]
            h_prev = out[:, : t + 1, :]
            h_q = h_t.unsqueeze(1).expand(-1, t + 1, -1)
            combined = torch.cat([h_q, h_prev], dim=-1)
            score = self.attn_v(torch.tanh(self.attn_W(combined))).squeeze(-1)
            valid = torch.arange(t + 1, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            score = score.masked_fill(~valid, -1e9)
            alpha = torch.softmax(score, dim=1)
            context = (alpha.unsqueeze(-1) * h_prev).sum(dim=1)
            out_t = torch.cat([h_t, context], dim=-1)
            logits_list.append(self.fc(out_t))
        return torch.stack(logits_list, dim=1)

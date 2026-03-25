import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Dict, List, Optional

from word2vec import (
    Word2Vec,
    build_vocab,
    compute_subsampling_keep_probs,
    make_noise_distribution,
    sentences_to_indices,
    tokenize,
    build_alias_table,
    sample_negative,
)


def _sigmoid(z: float) -> float:
    z = max(min(float(z), 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-z))


def _skipgram_pair_loss(vc: np.ndarray, uo: np.ndarray, u_negs: np.ndarray) -> float:
    z_pos = float(np.dot(vc, uo))
    s_pos = _sigmoid(z_pos)
    loss = -math.log(s_pos + 1e-10)
    if u_negs.size == 0:
        return loss
    z_negs = u_negs @ vc
    # -log sigma(-z) = -log(1 - sigma(z))
    s_negs = 1.0 / (1.0 + np.exp(-np.clip(z_negs, -35.0, 35.0)))
    loss += float(np.sum(-np.log(1.0 - s_negs + 1e-10)))
    return loss


def _cbow_example_loss(vc: np.ndarray, uo: np.ndarray, u_negs: np.ndarray) -> float:
    z_pos = float(np.dot(vc, uo))
    s_pos = _sigmoid(z_pos)
    loss = -math.log(s_pos + 1e-10)
    if u_negs.size == 0:
        return loss
    z_negs = u_negs @ vc
    s_negs = 1.0 / (1.0 + np.exp(-np.clip(z_negs, -35.0, 35.0)))
    loss += float(np.sum(-np.log(1.0 - s_negs + 1e-10)))
    return loss


def eval_skipgram(
    model: Word2Vec,
    indexed_sentences: List[List[int]],
    noise_prob: np.ndarray,
    *,
    window: int,
    num_negative: int,
    seed: int,
    subsample_keep_probs: Optional[np.ndarray],
) -> float:
    rng = random.Random(seed)
    q, j = build_alias_table(noise_prob)
    total_loss = 0.0
    total_pairs = 0
    for sent in indexed_sentences:
        if subsample_keep_probs is not None:
            sent = [idx for idx in sent if rng.random() < float(subsample_keep_probs[idx])]
        L = len(sent)
        if L < 2:
            continue
        for i, center in enumerate(sent):
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            vc = model.v_in[center]
            for jpos in range(lo, hi):
                if jpos == i:
                    continue
                context = sent[jpos]
                neg = sample_negative(rng, q, j, {center, context}, num_negative)
                uo = model.w_out[context]
                u_negs = model.w_out[neg] if neg else model.w_out[:0]
                total_loss += _skipgram_pair_loss(vc, uo, u_negs)
                total_pairs += 1
    if total_pairs == 0:
        return float("nan")
    return total_loss / total_pairs


def eval_cbow(
    model: Word2Vec,
    indexed_sentences: List[List[int]],
    noise_prob: np.ndarray,
    *,
    window: int,
    num_negative: int,
    seed: int,
    subsample_keep_probs: Optional[np.ndarray],
) -> float:
    rng = random.Random(seed)
    q, j = build_alias_table(noise_prob)
    total_loss = 0.0
    total_examples = 0
    for sent in indexed_sentences:
        if subsample_keep_probs is not None:
            sent = [idx for idx in sent if rng.random() < float(subsample_keep_probs[idx])]
        L = len(sent)
        if L < 2:
            continue
        for i, center in enumerate(sent):
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            ctx = [sent[jpos] for jpos in range(lo, hi) if jpos != i]
            if not ctx:
                continue
            vc = np.mean(model.v_in[ctx], axis=0)
            avoid = {center, *ctx}
            neg = sample_negative(rng, q, j, avoid, num_negative)
            uo = model.w_out[center]
            u_negs = model.w_out[neg] if neg else model.w_out[:0]
            total_loss += _cbow_example_loss(vc, uo, u_negs)
            total_examples += 1
    if total_examples == 0:
        return float("nan")
    return total_loss / total_examples


def _configure_pub_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _square_fig():
    return plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)


def _plot_loss_curves(out_path: Path, curves: Dict[str, List[float]], *, ylabel: str):
    _configure_pub_style()
    fig, ax = _square_fig()
    epochs = np.arange(1, len(next(iter(curves.values()))) + 1)
    for label, y in curves.items():
        ax.plot(epochs, y, marker="o", markersize=3.0, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(epochs)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
 


@dataclass(frozen=True)
class RunCfg:
    architecture: str  # "cbow" | "skipgram"
    window: int
    embedding_dim: int
    num_negative: int
    epochs: int
    learning_rate: float
    seed: int
    val_seed: int


def _train_and_validate(
    indexed_train: List[List[int]],
    indexed_val: List[List[int]],
    words: List[str],
    freqs: np.ndarray,
    *,
    cfg: RunCfg,
    subsample_t: float,
) -> List[float]:
    noise = make_noise_distribution(freqs)
    keep_probs = compute_subsampling_keep_probs(freqs, t=subsample_t)

    model = Word2Vec(len(words), cfg.embedding_dim, random.Random(cfg.seed))
    val_losses: List[float] = []

    for ep in range(cfg.epochs):
        if cfg.architecture == "skipgram":
            model.train_skipgram_sentences(
                indexed_train,
                noise,
                window=cfg.window,
                epochs=1,
                learning_rate=cfg.learning_rate,
                num_negative=cfg.num_negative,
                seed=cfg.seed + ep,
                subsample_keep_probs=keep_probs,
            )
            vloss = eval_skipgram(
                model,
                indexed_val,
                noise,
                window=cfg.window,
                num_negative=cfg.num_negative,
                seed=cfg.val_seed + ep,
                subsample_keep_probs=keep_probs,
            )
        elif cfg.architecture == "cbow":
            model.train_cbow_sentences(
                indexed_train,
                noise,
                window=cfg.window,
                epochs=1,
                learning_rate=cfg.learning_rate,
                num_negative=cfg.num_negative,
                seed=cfg.seed + ep,
                subsample_keep_probs=keep_probs,
            )
            vloss = eval_cbow(
                model,
                indexed_val,
                noise,
                window=cfg.window,
                num_negative=cfg.num_negative,
                seed=cfg.val_seed + ep,
                subsample_keep_probs=keep_probs,
            )
        else:
            raise ValueError(f"Unknown architecture: {cfg.architecture}")

        val_losses.append(float(vloss))

    return val_losses


def _split_train_val(indexed: List[List[int]], *, val_ratio: float, seed: int):
    rng = random.Random(seed)
    idxs = list(range(len(indexed)))
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(indexed) * val_ratio)))
    val_set = [indexed[i] for i in idxs[:n_val]]
    train_set = [indexed[i] for i in idxs[n_val:]]
    if not train_set:
        raise ValueError("Train split ended up empty. Reduce val_ratio.")
    return train_set, val_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(Path(__file__).resolve().parent / "corpus.txt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--subsample-t", type=float, default=1e-5)
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "plots"))
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    sents = [tokenize(line) for line in text.splitlines() if line.strip()]
    word2idx, words, freqs = build_vocab(sents)
    indexed = sentences_to_indices(sents, word2idx)
    indexed_train, indexed_val = _split_train_val(indexed, val_ratio=float(args.val_ratio), seed=int(args.split_seed))

    base_dim = 300
    base_window = 5
    base_neg = 5
    lr_sg = 0.1
    lr_cb = 0.15

    # i) sweep window: 1, 3, 5
    for arch, lr in (("cbow", lr_cb), ("skipgram", lr_sg)):
        curves = {}
        for w in (1, 3, 5):
            cfg = RunCfg(
                architecture=arch,
                window=w,
                embedding_dim=base_dim,
                num_negative=base_neg,
                epochs=int(args.epochs),
                learning_rate=lr,
                seed=1000 + w,
                val_seed=2000 + w,
            )
            curves[f"w={w}"] = _train_and_validate(
                indexed_train,
                indexed_val,
                words,
                freqs,
                cfg=cfg,
                subsample_t=float(args.subsample_t),
            )
        _plot_loss_curves(outdir / f"{arch}_val_loss_sweep_window.png", curves, ylabel="Validation loss")

    # ii) sweep embedding dim: 100, 200, 300
    for arch, lr in (("cbow", lr_cb), ("skipgram", lr_sg)):
        curves = {}
        for d in (100, 300, 500):
            cfg = RunCfg(
                architecture=arch,
                window=base_window,
                embedding_dim=d,
                num_negative=base_neg,
                epochs=int(args.epochs),
                learning_rate=lr,
                seed=3000 + d,
                val_seed=4000 + d,
            )
            curves[f"d={d}"] = _train_and_validate(
                indexed_train,
                indexed_val,
                words,
                freqs,
                cfg=cfg,
                subsample_t=float(args.subsample_t),
            )
        _plot_loss_curves(outdir / f"{arch}_val_loss_sweep_dim.png", curves, ylabel="Validation loss")

    # iii) sweep negative samples: 1, 3, 5
    for arch, lr in (("cbow", lr_cb), ("skipgram", lr_sg)):
        curves = {}
        for k in (1, 3, 5):
            cfg = RunCfg(
                architecture=arch,
                window=base_window,
                embedding_dim=base_dim,
                num_negative=k,
                epochs=int(args.epochs),
                learning_rate=lr,
                seed=5000 + k,
                val_seed=6000 + k,
            )
            curves[f"k={k}"] = _train_and_validate(
                indexed_train,
                indexed_val,
                words,
                freqs,
                cfg=cfg,
                subsample_t=float(args.subsample_t),
            )
        _plot_loss_curves(outdir / f"{arch}_val_loss_sweep_neg.png", curves, ylabel="Validation loss")


if __name__ == "__main__":
    main()


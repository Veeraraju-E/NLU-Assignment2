import argparse
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from word2vec import Word2Vec, tokenize
from evaluate_word2vec import resolve_checkpoint_dir


def _select_words_from_corpus(corpus_path: Path, *, top_n: int) -> List[str]:
    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    counts = Counter(tokenize(text))
    return [w for (w, _c) in counts.most_common(top_n)]


def _project_2d(vectors: np.ndarray, *, method: str, seed: int) -> np.ndarray:
    method = method.lower().strip()
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(vectors)

    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=seed,
        ).fit_transform(vectors)

    raise ValueError(f"Unknown projection method: {method}. Use 'pca' or 'tsne'.")


def _cluster_labels(vectors: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    if k <= 0:
        return np.full((vectors.shape[0],), -1, dtype=np.int64)

    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init="auto", random_state=seed).fit_predict(vectors).astype(np.int64)


def visualize_one(
    checkpoint_dir: Path,
    *,
    corpus_path: Path,
    method: str,
    top_n: int,
    k: int,
    seed: int,
    output_path: Path,
    annotate_max: int,
):
    import matplotlib.pyplot as plt

    model, vocab, word2idx, metadata = Word2Vec.load_checkpoint(checkpoint_dir)
    embeddings = model.get_combined_embeddings()

    selected = _select_words_from_corpus(corpus_path, top_n=top_n)
    selected = [w for w in selected if w in word2idx]
    if not selected:
        raise ValueError("No selected words were found in the checkpoint vocabulary.")

    idxs = np.array([word2idx[w] for w in selected], dtype=np.int64)
    vecs = embeddings[idxs]
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

    xy = _project_2d(vecs, method=method, seed=seed)
    labels = _cluster_labels(vecs, k=k, seed=seed)

    title_arch = metadata.get("architecture", checkpoint_dir.name)
    title = f"Word2Vec {title_arch} ({method.upper()}) | words={len(selected)}"

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    if k > 0:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=24, cmap="tab10", alpha=0.85, linewidths=0.0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("KMeans cluster id")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=24, alpha=0.85, linewidths=0.0)

    if annotate_max > 0:
        n_annot = min(len(selected), int(annotate_max))
        for i in range(n_annot):
            ax.annotate(
                selected[i],
                (float(xy[i, 0]), float(xy[i, 1])),
                fontsize=7,
                alpha=0.9,
                xytext=(2, 2),
                textcoords="offset points",
            )

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Word2Vec Skip-gram / CBOW clusters using PCA or t-SNE projection."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="C:/Users/Veeraraju_elluru/Desktop/Veeraraju/IITJ/Senior_Year/Sem8/NLU/Assignment-2/part1",
        help="Checkpoint path or project root (expects checkpoints/{skipgram,cbow})",
    )
    parser.add_argument(
        "--architecture",
        choices=("skipgram", "cbow", "both"),
        default="both",
        help="Which model checkpoints to visualize",
    )
    parser.add_argument("--method", choices=("pca", "tsne"), default="pca", help="2D projection method")
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Pick top-N frequent corpus words to visualize (filtered by vocab)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="KMeans cluster count for coloring; set to 0 to disable clustering",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for projection/clustering")
    parser.add_argument(
        "--annotate-max",
        type=int,
        default=80,
        help="Max number of point labels to draw (reduce if plot is cluttered)",
    )
    parser.add_argument(
        "--corpus-path",
        default="",
        help="Optional corpus.txt path (defaults to part1/corpus.txt next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory (defaults to part1/plots)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    checkpoint_root = Path(args.checkpoint_dir)
    corpus_path = Path(args.corpus_path) if args.corpus_path else (base_dir / "corpus.txt")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at: {corpus_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "plots")

    architectures = ("skipgram", "cbow") if args.architecture == "both" else (args.architecture,)
    for arch in architectures:
        ckpt = resolve_checkpoint_dir(checkpoint_root, arch)
        out = output_dir / f"clusters_{arch}_{args.method}.png"
        visualize_one(
            ckpt,
            corpus_path=corpus_path,
            method=args.method,
            top_n=int(args.top_n),
            k=int(args.clusters),
            seed=int(args.seed),
            output_path=out,
            annotate_max=int(args.annotate_max),
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np

from word2vec import Word2Vec


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def nearest_neighbors(query_word, normalized_embeddings, words, word2idx, top_k=5):
    if query_word not in word2idx:
        raise ValueError(f"Word '{query_word}' is not in vocabulary.")
    q_idx = word2idx[query_word]
    sims = normalized_embeddings @ normalized_embeddings[q_idx]
    sims[q_idx] = -np.inf
    best = np.argpartition(-sims, top_k)[:top_k]
    best = best[np.argsort(-sims[best])]
    return [(words[i], float(sims[i])) for i in best]


def analogy(a, b, c, normalized_embeddings, words, word2idx, top_k=5):
    missing = [w for w in (a, b, c) if w not in word2idx]
    if missing:
        raise ValueError(f"Analogy words missing from vocabulary: {missing}")
    va = normalized_embeddings[word2idx[a]]
    vb = normalized_embeddings[word2idx[b]]
    vc = normalized_embeddings[word2idx[c]]
    target = vb - va + vc
    target_norm = np.linalg.norm(target)
    if target_norm == 0.0:
        raise ValueError("Analogy target vector became zero.")
    target = target / target_norm

    sims = normalized_embeddings @ target
    for w in (a, b, c):
        sims[word2idx[w]] = -np.inf
    best = np.argpartition(-sims, top_k)[:top_k]
    best = best[np.argsort(-sims[best])]
    return [(words[i], float(sims[i])) for i in best]


def resolve_checkpoint_dir(path, architecture):
    checkpoint_dir = Path(path)
    required = ("v_in.npy", "w_out.npy", "vocab.txt", "metadata.json")
    if all((checkpoint_dir / name).exists() for name in required):
        return checkpoint_dir

    candidate = checkpoint_dir / "checkpoints" / architecture
    if all((candidate / name).exists() for name in required):
        return candidate

    candidate = checkpoint_dir / architecture
    if all((candidate / name).exists() for name in required):
        return candidate

    raise FileNotFoundError(
        f"Could not find checkpoint files in '{checkpoint_dir}'. "
        f"Tried '{checkpoint_dir}', '{checkpoint_dir / 'checkpoints' / architecture}', "
        f"and '{checkpoint_dir / architecture}'."
    )


def write_student_embedding(checkpoint_dir, architecture, embeddings, word2idx):
    if "student" not in word2idx:
        raise ValueError("Word 'student' is not in vocabulary.")
    vec = embeddings[word2idx["student"]]
    values = ", ".join(f"{x:.4f}" for x in vec)
    out_path = Path(checkpoint_dir) / f"{architecture}_student_embedding.txt"
    out_path.write_text(f"Student- {values}\n", encoding="utf-8")
    print(f"\nSaved student embedding to: {out_path}")


def evaluate_one(checkpoint_dir, *, top_k):
    model, words, word2idx, metadata = Word2Vec.load_checkpoint(checkpoint_dir)
    embeddings = model.get_combined_embeddings()
    normalized = normalize_rows(embeddings)
    architecture = metadata.get("architecture", "unknown")

    print(f"Loaded checkpoint: {checkpoint_dir}")
    print(f"Architecture: {architecture}")
    print(f"Vocab size: {len(words)} | Embedding dim: {embeddings.shape[1]}\n")

    query_words = ["research", "student", "phd", "examination", "director"]
    print("Top nearest neighbors using cosine similarity")
    for w in query_words:
        print(f"\n{w}:")
        for rank, (neighbor, score) in enumerate(
            nearest_neighbors(w, normalized, words, word2idx, top_k=top_k),
            start=1,
        ):
            print(f"  {rank}. {neighbor:20s} cos={score:.6f}")

    experiments = [
        ("semester", "sgpa", "degree"),
        ("course", "grade", "thesis"),
        ("institute", "director", "senate"),
    ]
    print("\nAnalogy experiments (a : b :: c : ?)")
    for a, b, c in experiments:
        print(f"\n{a} : {b} :: {c} : ?")
        for rank, (candidate, score) in enumerate(
            analogy(a, b, c, normalized, words, word2idx, top_k=top_k),
            start=1,
        ):
            print(f"  {rank}. {candidate:20s} cos={score:.6f}")

    write_student_embedding(checkpoint_dir, architecture, embeddings, word2idx)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Word2Vec checkpoints.")
    parser.add_argument("--checkpoint-dir", default="C:/Users/Veeraraju_elluru/Desktop/Veeraraju/IITJ/Senior_Year/Sem8/NLU/Assignment-2/part1", help="Checkpoint path or project root created by word2vec.py")
    parser.add_argument("--architecture", choices=("skipgram", "cbow", "both"), default="both", help="Architecture to evaluate when checkpoint-dir points to a parent directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matches to print")
    args = parser.parse_args()

    if args.architecture == "both":
        for arch in ("skipgram", "cbow"):
            print(f"\n{'=' * 18} {arch} {'=' * 18}\n")
            checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, arch)
            evaluate_one(checkpoint_dir, top_k=args.top_k)
    else:
        checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.architecture)
        evaluate_one(checkpoint_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()

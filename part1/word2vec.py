# imports
import math
import random
import re
from collections import Counter
import json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# utilities/helpers
def sigmoid(z):
    z = max(min(z, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-z))


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def build_vocab(sentences):
    counts = Counter()
    for sent in sentences:
        counts.update(sent)
    words = [w for w, c in counts.items()]
    words.sort(key=lambda w: (-counts[w], w))
    word2idx = {w: i for i, w in enumerate(words)}
    freqs = np.array([counts[w] for w in words], dtype=np.float64)
    return word2idx, words, freqs


def sentences_to_indices(sentences, word2idx, min_length = 2):
    out = []
    for sent in sentences:
        idxs = [word2idx[w] for w in sent if w in word2idx]
        if len(idxs) >= min_length:
            out.append(idxs)
    return out


def make_noise_distribution(freqs, power = 0.75):
    p = np.power(freqs, power)
    p /= p.sum()
    return p


def compute_subsampling_keep_probs(freqs, t=1e-5):
    """
    Word2Vec subsampling (Mikolov et al., 2013).

    Given raw token counts (freqs), returns per-word keep probabilities in [0,1].
    Keep probability p_keep(w) = min(1, (sqrt(f/t) + 1) * (t/f)),
    where f is the empirical frequency (count / total_count).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    total = freqs.sum()
    if total <= 0:
        raise ValueError("Total token count must be positive for subsampling.")
    f = freqs / total
    # Avoid division by zero for any degenerate counts.
    f = np.maximum(f, 1e-18)
    p_keep = (np.sqrt(f / t) + 1.0) * (t / f)
    return np.clip(p_keep, 0.0, 1.0)


def build_alias_table(prob):
    """Vose alias method for O(1) sampling from a discrete distribution."""
    n = len(prob)
    prob = prob.astype(np.float64) * n
    small = []
    large = []
    q = np.zeros(n, dtype=np.float64)
    j = np.zeros(n, dtype=np.int64)
    for i in range(n):
        q[i] = prob[i]
        if q[i] < 1.0:
            small.append(i)
        else:
            large.append(i)
    while small and large:
        s = small.pop()
        l = large.pop()
        j[s] = l
        q[l] = q[l] + q[s] - 1.0
        if q[l] < 1.0:
            small.append(l)
        else:
            large.append(l)
    while large:
        l = large.pop()
        q[l] = 1.0
    while small:
        s = small.pop()
        q[s] = 1.0
    return q, j


def sample_negative(rng, q, j, avoid, n_samples):
    out = []
    nv = len(q)
    while len(out) < n_samples:
        i = rng.randrange(nv)
        u = rng.random()
        idx = i if u < q[i] else int(j[i])
        if idx not in avoid:
            out.append(idx)
    return out

# training functions
def train_skipgram_step(v_in, w_out, center, context, neg_indices, lr):
    """
    One skip-gram pair (center predicts context). Returns scalar loss contribution.
    Positive: minimize -log σ(z)  =>  g_pos = σ(z) - 1.
    Negative k: minimize -log σ(-z_k)  =>  g_k = σ(z_k).
    """
    vc = v_in[center]
    loss = 0.0

    u_o = w_out[context]
    z_pos = float(np.dot(vc, u_o))
    s_pos = sigmoid(z_pos)
    loss += -math.log(s_pos + 1e-10)
    g_pos = s_pos - 1.0

    grad_vc = g_pos * u_o
    w_out[context] -= lr * g_pos * vc

    for k in neg_indices:
        u_k = w_out[k]
        z_k = float(np.dot(vc, u_k))
        s_k = sigmoid(z_k)
        loss += -math.log(1.0 - s_k + 1e-10)
        g_k = s_k
        grad_vc += g_k * u_k
        w_out[k] -= lr * g_k * vc

    v_in[center] -= lr * grad_vc
    return loss


def train_cbow_step(v_in, w_out, context_idxs, center, neg_indices, lr):
    """
    CBOW: average of context input vectors predicts center via output embeddings.
    """
    n_ctx = len(context_idxs)
    if n_ctx == 0:
        return 0.0
    vc = np.mean(v_in[context_idxs], axis=0)
    loss = 0.0

    u_o = w_out[center]
    z_pos = float(np.dot(vc, u_o))
    s_pos = sigmoid(z_pos)
    loss += -math.log(s_pos + 1e-10)
    g_pos = s_pos - 1.0

    grad_vc = g_pos * u_o
    w_out[center] -= lr * g_pos * vc

    for k in neg_indices:
        u_k = w_out[k]
        z_k = float(np.dot(vc, u_k))
        s_k = sigmoid(z_k)
        loss += -math.log(1.0 - s_k + 1e-10)
        g_k = s_k
        grad_vc += g_k * u_k
        w_out[k] -= lr * g_k * vc

    grad_per_ctx = (lr / n_ctx) * grad_vc
    for cidx in context_idxs:
        v_in[cidx] -= grad_per_ctx

    return loss

# main class
class Word2Vec:
    """
    Word2Vec with negative sampling.
    """

    def __init__(self, vocab_size, embedding_dim, rng=None):
        self.rng = rng or random.Random()
        bound = 0.5 / embedding_dim
        init_seed = self.rng.randint(0, 2**31 - 1)
        gen = np.random.default_rng(init_seed)
        self.v_in = gen.uniform(-bound, bound, size=(vocab_size, embedding_dim)).astype(np.float64)
        self.w_out = gen.uniform(-bound, bound, size=(vocab_size, embedding_dim)).astype(np.float64)

    def get_combined_embeddings(self):
        """Return standard word2vec embedding estimate: input + output vectors."""
        return self.v_in + self.w_out

    def save_checkpoint(self, checkpoint_dir, words, *, metadata=None):
        """
        Save a reusable checkpoint with:
        - input embeddings
        - output embeddings
        - combined embeddings (for downstream cosine similarity tasks)
        - vocabulary list
        - metadata json
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        np.save(checkpoint_dir / "v_in.npy", self.v_in)
        np.save(checkpoint_dir / "w_out.npy", self.w_out)
        np.save(checkpoint_dir / "embeddings.npy", self.get_combined_embeddings())
        (checkpoint_dir / "vocab.txt").write_text("\n".join(words), encoding="utf-8")

        meta = metadata.copy() if metadata else {}
        meta["vocab_size"] = int(self.v_in.shape[0])
        meta["embedding_dim"] = int(self.v_in.shape[1])
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(cls, checkpoint_dir):
        """
        Load a checkpoint created by save_checkpoint.
        Returns (model, words, word2idx, metadata).
        """
        checkpoint_dir = Path(checkpoint_dir)
        v_in = np.load(checkpoint_dir / "v_in.npy")
        w_out = np.load(checkpoint_dir / "w_out.npy")
        words = (checkpoint_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))

        if v_in.shape != w_out.shape:
            raise ValueError("Checkpoint embeddings shape mismatch between v_in and w_out.")
        if len(words) != v_in.shape[0]:
            raise ValueError("Checkpoint vocab size does not match embedding rows.")

        model = cls(vocab_size=v_in.shape[0], embedding_dim=v_in.shape[1], rng=random.Random(0))
        model.v_in = v_in
        model.w_out = w_out
        word2idx = {w: i for i, w in enumerate(words)}
        return model, words, word2idx, metadata

    def train_skipgram_sentences(
        self,
        indexed_sentences,
        noise_prob,
        *,
        window,
        epochs,
        learning_rate,
        num_negative,
        seed=42,
        subsample_keep_probs=None,
        min_lr_ratio=1e-4,
        record_step_loss=False,
    ):
        """
        Skip-gram training that generates (center, context) pairs on-the-fly each epoch.
        This matches the original word2vec training flow more closely than precomputing pairs.
        """
        rng = random.Random(seed)
        q, j = build_alias_table(noise_prob)
        lr0 = float(learning_rate)
        min_lr = lr0 * float(min_lr_ratio)

        # Pre-compute an epoch step estimate for LR decay (exact count depends on dynamic window + subsampling).
        est_pairs_per_tok = float(window)
        est_steps = 0
        for sent in indexed_sentences:
            est_steps += int(len(sent) * est_pairs_per_tok)
        if est_steps <= 0:
            if record_step_loss:
                return [], []
            return []
        total_steps = int(epochs) * int(est_steps)

        step_losses = [] if record_step_loss else None
        avg_losses = []
        global_step = 0

        epoch_bar = tqdm(range(epochs), desc="skip-gram", unit="epoch")
        for ep in epoch_bar:
            epoch_loss = 0.0
            epoch_steps = 0
            with tqdm(total=est_steps, desc=f"sg ep {ep + 1}/{epochs}", leave=False, unit="pair", mininterval=0.0, miniters=1) as pair_bar:
                for sent in indexed_sentences:
                    if subsample_keep_probs is not None:
                        sent = [idx for idx in sent if rng.random() < float(subsample_keep_probs[idx])]
                    L = len(sent)
                    if L < 2:
                        continue
                    for i, center in enumerate(sent):
                        lo = max(0, i - window)
                        hi = min(L, i + window + 1)
                        for jpos in range(lo, hi):
                            if jpos == i:
                                continue
                            context = sent[jpos]
                            global_step += 1
                            lr = lr0 * (1.0 - (global_step / total_steps))
                            if lr < min_lr:
                                lr = min_lr
                            neg = sample_negative(rng, q, j, {center, context}, num_negative)
                            ell = train_skipgram_step(self.v_in, self.w_out, center, context, neg, lr)
                            epoch_loss += ell
                            epoch_steps += 1
                            if record_step_loss:
                                step_losses.append(ell)
                            if epoch_steps % 50 == 0:
                                pair_bar.update(50)
                # Flush any remaining progress for nicer bars
                rem = epoch_steps - pair_bar.n
                if rem > 0:
                    pair_bar.update(rem)
            if epoch_steps == 0:
                avg = float("nan")
            else:
                avg = epoch_loss / epoch_steps
            avg_losses.append(avg)
            epoch_bar.set_postfix(avg_loss=f"{avg:.6f}", last_ep=f"{ep + 1}/{epochs}")
        if record_step_loss:
            return avg_losses, step_losses
        return avg_losses

    def train_cbow_sentences(
        self,
        indexed_sentences,
        noise_prob,
        *,
        window,
        epochs,
        learning_rate,
        num_negative,
        seed=42,
        subsample_keep_probs=None,
        min_lr_ratio=1e-4,
        record_step_loss=False,
    ):
        """
        CBOW training that generates (context, center) examples on-the-fly each epoch.
        This is closer to the original word2vec training loop and usually converges better than
        using a fixed precomputed example list.
        """
        rng = random.Random(seed)
        q, j = build_alias_table(noise_prob)
        lr0 = float(learning_rate)
        min_lr = lr0 * float(min_lr_ratio)

        est_examples_per_tok = 1.0
        est_steps = 0
        for sent in indexed_sentences:
            est_steps += int(len(sent) * est_examples_per_tok)
        if est_steps <= 0:
            if record_step_loss:
                return [], []
            return []
        total_steps = int(epochs) * int(est_steps)

        step_losses = [] if record_step_loss else None
        avg_losses = []
        global_step = 0

        epoch_bar = tqdm(range(epochs), desc="cbow", unit="epoch")
        for ep in epoch_bar:
            epoch_loss = 0.0
            epoch_steps = 0
            with tqdm(total=est_steps, desc=f"cbow ep {ep + 1}/{epochs}", leave=False, unit="ex", mininterval=0.0, miniters=1) as ex_bar:
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
                        global_step += 1
                        lr = lr0 * (1.0 - (global_step / total_steps))
                        if lr < min_lr:
                            lr = min_lr
                        avoid = {center, *ctx}
                        neg = sample_negative(rng, q, j, avoid, num_negative)
                        ell = train_cbow_step(self.v_in, self.w_out, ctx, center, neg, lr)
                        epoch_loss += ell
                        epoch_steps += 1
                        if record_step_loss:
                            step_losses.append(ell)
                        if epoch_steps % 50 == 0:
                            ex_bar.update(50)
                rem = epoch_steps - ex_bar.n
                if rem > 0:
                    ex_bar.update(rem)
            if epoch_steps == 0:
                avg = float("nan")
            else:
                avg = epoch_loss / epoch_steps
            avg_losses.append(avg)
            epoch_bar.set_postfix(avg_loss=f"{avg:.6f}", last_ep=f"{ep + 1}/{epochs}")
        if record_step_loss:
            return avg_losses, step_losses
        return avg_losses


def main():
    base = Path(__file__).resolve().parent
    corpus_path = base / "corpus.txt"
    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    sents = [tokenize(line) for line in text.splitlines() if line.strip()]
    word2idx, words, freqs = build_vocab(sents)
    indexed = sentences_to_indices(sents, word2idx)
    noise = make_noise_distribution(freqs)

    # hyperparameters
    dim = 300
    window = 5
    neg = 5
    lr_skipgram = 0.1
    lr_cbow = 0.15
    epochs = 25
    subsample_t = 1e-5

    # Word2vec subsampling keep probabilities.
    keep_probs = compute_subsampling_keep_probs(freqs, t=subsample_t)

    model_sg = Word2Vec(len(words), dim, random.Random(0))
    model_sg.train_skipgram_sentences(
        indexed,
        noise,
        window=window,
        epochs=epochs,
        learning_rate=lr_skipgram,
        num_negative=neg,
        seed=42,
        subsample_keep_probs=keep_probs,
    )
    model_sg.save_checkpoint(
        base / "checkpoints" / "skipgram",
        words,
        metadata={
            "architecture": "skipgram",
            "window": window,
            "num_negative": neg,
            "learning_rate": lr_skipgram,
            "epochs": epochs,
            "corpus": str(corpus_path.name),
            "subsample_t": subsample_t,
        },
    )
    print("skip-gram training done")

    model_cb = Word2Vec(len(words), dim, random.Random(1))
    model_cb.train_cbow_sentences(
        indexed,
        noise,
        window=window,
        epochs=epochs,
        learning_rate=lr_cbow,
        num_negative=neg,
        seed=43,
        subsample_keep_probs=keep_probs,
    )
    model_cb.save_checkpoint(
        base / "checkpoints" / "cbow",
        words,
        metadata={
            "architecture": "cbow",
            "window": window,
            "num_negative": neg,
            "learning_rate": lr_cbow,
            "epochs": epochs,
            "corpus": str(corpus_path.name),
            "subsample_t": subsample_t,
        },
    )
    print("cbow training done")


if __name__ == "__main__":
    main()

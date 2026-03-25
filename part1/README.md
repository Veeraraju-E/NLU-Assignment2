# Part 1 - Word2Vec on IITJ Corpus

This part implements Word2Vec from scratch (NumPy) with:

- Skip-gram with negative sampling
- CBOW with negative sampling
- Word subsampling (Mikolov et al., 2013 style)
- Checkpoint save/load for reproducible evaluation

The training corpus is `corpus.txt`.

## Files

- `word2vec.py`: core training implementation and checkpoint export.
- `loss_sweeps.py`: sweeps context window, embedding dimension, and negative samples.
- `evaluate_word2vec.py`: nearest-neighbor and analogy evaluation.
- `visualize_word2vec_clusters.py`: PCA/t-SNE projection with optional KMeans clusters.
- `dataset_stats.py`: token stats and word cloud generation.
- `requirements.txt`: Python dependencies for this part.

## Setup

From project root:

```bash
pip install -r part1/requirements.txt
```

## 1) Train Skip-gram and CBOW

```bash
python part1/word2vec.py
```

This creates:

- `part1/checkpoints/skipgram/`
- `part1/checkpoints/cbow/`

Each checkpoint contains:

- `v_in.npy`, `w_out.npy`, `embeddings.npy`
- `vocab.txt`
- `metadata.json`

## 2) Evaluate Embeddings

Evaluate both models:

```bash
python part1/evaluate_word2vec.py --checkpoint-dir part1 --architecture both --top-k 5
```

Evaluate only one model:

```bash
python part1/evaluate_word2vec.py --checkpoint-dir part1 --architecture skipgram
```

The script prints:

- cosine-similarity nearest neighbors for selected query words
- analogy results (`a : b :: c : ?`)

## 3) Hyperparameter Sweeps and Loss Plots

```bash
python part1/loss_sweeps.py --epochs 10 --outdir part1/plots
```

Generated files:

- `part1/plots/cbow_val_loss_sweep_window.png`
- `part1/plots/skipgram_val_loss_sweep_window.png`
- `part1/plots/cbow_val_loss_sweep_dim.png`
- `part1/plots/skipgram_val_loss_sweep_dim.png`
- `part1/plots/cbow_val_loss_sweep_neg.png`
- `part1/plots/skipgram_val_loss_sweep_neg.png`

## 4) Embedding Space Visualization

```bash
python part1/visualize_word2vec_clusters.py --checkpoint-dir part1 --architecture both --method pca --top-n 200 --clusters 8
```

Outputs (default):

- `part1/plots/clusters_skipgram_pca.png`
- `part1/plots/clusters_cbow_pca.png`

Use `--method tsne` for t-SNE.

## 5) Corpus Stats + Word Cloud

```bash
python part1/dataset_stats.py
```

Outputs:

- token and vocabulary counts in terminal
- `part1/corpus_wordcloud.png`

## Method/Library Credits

- Word2Vec training design is based on:
  - Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
  - Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (2013)
- Uses standard libraries: NumPy, tqdm, Matplotlib, scikit-learn, WordCloud.

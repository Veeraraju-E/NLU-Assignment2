# Natural Language Understanding - Assignment 2

This repository implements two core NLP pipelines end-to-end:

1. **Distributional semantics (Word2Vec)** on a curated IITJ academic corpus.
2. **Character-level sequence generation** for Indian names using recurrent neural networks.

The project is structured to support both **training from scratch** and **reproducible analysis** (evaluation scripts, sweep scripts, checkpoints, and report-ready outputs).

## What The Repository Does

### Part 1 (`part1/`) - Word2Vec

- Implements Skip-gram and CBOW with negative sampling in NumPy.
- Learns embeddings from `part1/corpus.txt`.
- Exports reusable checkpoints (vocabulary + input/output embeddings + metadata).
- Evaluates semantic quality with nearest-neighbor and analogy queries.
- Produces sensitivity plots for key hyperparameters (window size, embedding dimension, negatives).
- Visualizes learned embedding geometry via PCA/t-SNE + KMeans.

### Part 2 (`part2/`) - Character-Level Name Generation

- Builds a character vocabulary from `part2/TrainingNames.txt`.
- Trains next-character predictors using:
  - `vanilla_rnn`
  - `blstm`
  - `rnn_attention`
- Supports checkpointing, resume training, and sample generation.
- Includes grid-sweep scripts and stored CSV sweep outputs.
- Includes generated sample files for qualitative comparison across model classes.

### Report (`paper/`)

- Contains the LaTeX report with methodology, experimental setup, and findings.

## Repository Layout

- `part1/word2vec.py`: Word2Vec training and checkpoint writing.
- `part1/evaluate_word2vec.py`: nearest-neighbor + analogy evaluation.
- `part1/loss_sweeps.py`: validation-loss sweeps and plot generation.
- `part1/visualize_word2vec_clusters.py`: embedding projection and clustering visualizations.
- `part2/main.py`: model training, validation, checkpointing, and sample generation.
- `part2/hyperparam.py`: grid sweep for sequence-model hyperparameters.
- `part2/model.py`, `part2/train.py`, `part2/dataloader.py`: model definitions and training/data pipeline.
- `paper/main.tex`: assignment report source.

## Setup

Use Python 3.10+.

```bash
pip install -r part1/requirements.txt
pip install -r part2/requirements.txt
```

## Minimal Reproduction

From the repository root:

```bash
# Part 1: train Word2Vec (skip-gram + CBOW)
python part1/word2vec.py

# Part 1: evaluate checkpoints
python part1/evaluate_word2vec.py --checkpoint-dir part1 --architecture both

# Part 2: train name generator (example)
python part2/main.py --model vanilla_rnn --data part2/TrainingNames.txt
```

For deeper analyses, see `part1/README.md` and `part2/README.md`.

## Outputs And Artifacts

- **Part 1 outputs:** `part1/checkpoints/`, `part1/plots/`, corpus word cloud.
- **Part 2 outputs:** `part2/checkpoints/`, `part2/hyperparam_sweep_results*.csv`, `part2/*_out.txt`.
- Scripts are path-safe from project root and expose seed/config arguments for reproducibility.

## Methods And Credits

- Word2Vec objectives, negative sampling, and subsampling follow Mikolov et al. (2013).
- Sequence models and training rely on standard PyTorch recurrent modules and optimizers.
- Analysis/visualization uses NumPy, Matplotlib, scikit-learn, tqdm, and WordCloud.

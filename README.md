# Natural Language Understanding - Assignment 2

This repository contains two implemented parts for NLU coursework:

- `part1/`: Word2Vec (Skip-gram and CBOW) training and analysis on an IITJ-focused corpus.
- `part2/`: Character-level Indian name generation using `VanillaRNN`, `BLSTM`, and `RNNWithAttention`.

## Repository Structure

- `part1/word2vec.py`: core Word2Vec implementation (negative sampling, subsampling, checkpoints).
- `part1/loss_sweeps.py`: hyperparameter sweeps and validation-loss plots.
- `part1/evaluate_word2vec.py`: nearest-neighbor and analogy evaluation.
- `part1/visualize_word2vec_clusters.py`: PCA/t-SNE embedding visualization with KMeans coloring.
- `part2/main.py`: training + checkpointing + generation entrypoint.
- `part2/hyperparam.py`: hyperparameter sweep for sequence models.
- `part2/model.py`, `part2/train.py`, `part2/dataloader.py`: model/data/training modules.

## Quick Start

Use Python 3.10+ (recommended) and install dependencies per part:

```bash
pip install -r part1/requirements.txt
pip install -r part2/requirements.txt
```

Then run:

```bash
python part1/word2vec.py
python part2/main.py --model vanilla_rnn
```

## Reproducibility Notes

- Scripts expose random seeds through arguments (for controlled reruns).
- Paths are written to work from project root.
- Checkpoints and plots are written inside each part folder.

## Credits

This implementation follows established methods and libraries:

- Word2Vec objective + negative sampling + subsampling: Mikolov et al. (2013).
- Sequence modeling stack: PyTorch recurrent modules and training utilities.
- Visualization and clustering: Matplotlib, scikit-learn (PCA/t-SNE/KMeans), WordCloud.

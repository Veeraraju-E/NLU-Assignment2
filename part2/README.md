# Part 2 - Character-Level Name Generation

This part trains sequence models to generate Indian names from `TrainingNames.txt`.

Implemented models:

- `vanilla_rnn`
- `blstm`
- `rnn_attention`

Core implementation files:

- `main.py`: train, validate, checkpoint, and generate names.
- `hyperparam.py`: grid sweep over model hyperparameters.
- `model.py`: model definitions.
- `train.py`: training/evaluation loops and generation function.
- `dataloader.py`: dataset loading, vocabulary building, and train/val/test splits.

## Setup

From project root:

```bash
pip install -r part2/requirements.txt
```

## Train a Model

Example (vanilla RNN):

```bash
python part2/main.py --model vanilla_rnn --data part2/TrainingNames.txt --epochs 50 --batch_size 128 --embed 256 --hidden 256 --layers 2 --lr 0.001
```

Other options:

```bash
python part2/main.py --model blstm
python part2/main.py --model rnn_attention
```

By default, `main.py`:

- splits data into train/val/test with ratios 0.7/0.2/0.1
- resumes from latest checkpoint unless `--no_resume` is set
- saves checkpoints under `part2/checkpoints/`
- prints generated samples after training

Checkpoint files:

- `part2/checkpoints/<model>_best.pt`
- `part2/checkpoints/<model>_latest.pt`

## Hyperparameter Sweep

Run a sweep:

```bash
python part2/hyperparam.py --model vanilla_rnn --epochs 50 --embed_range 128,256 --hidden_range 128,256 --layers_range 2,5,8 --dropout_range 0.1,0.2,0.3 --lr_range 0.0001,0.001 --batch_range 128 --results_csv part2/hyperparam_sweep_results_vanilla.csv
```

For attention model:

```bash
python part2/hyperparam.py --model rnn_attention --results_csv part2/hyperparam_sweep_results.csv
```

Existing result files:

- `hyperparam_sweep_results.csv` (attention runs)
- `hyperparam_sweep_results_vanilla.csv` (vanilla RNN runs)

## Included Sample Outputs

- `rnn_out.txt`: generated names from vanilla RNN run.
- `rnn_attention_out.txt`: generated names from attention RNN run.
- `bilstm_out.txt`: generated names from BLSTM run.

The current outputs indicate:

- vanilla RNN generates the highest proportion of names matching corpus entries,
- attention model produces partially realistic names,
- BLSTM output quality is weakest in this setup.

## Method/Library Credits

- Character sequence modeling built with PyTorch RNN/LSTM modules.
- Training/evaluation routines use standard teacher-forcing next-character prediction with cross-entropy.
- Progress reporting uses tqdm.

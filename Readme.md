# Federated Learning with Differential Privacy

This project implements **CorBin-FL** and **Augmented CorBin-FL** — differentially private federated learning methods based on correlated binary stochastic quantization — and compares them against a broad set of LDP, CDP, and non-private baselines across multiple datasets.

## Features

### FL Methods

| Method | Key | Privacy | Bits/coord | Reference |
|--------|-----|---------|------------|-----------|
| CorBin-FL | `CorbinFL` | ε-LDP | 1 | Salehi et al. 2026 (Ours) |
| Augmented CorBin-FL | `AugCorbinFL` | ε-LDP | 1 | Salehi et al. 2026 (Ours) |
| LDP-FL | `LDPFL` | ε-LDP | 1 | Sun et al. 2020 |
| I-MVU | `IMVU` | ε-LDP | 1 / 2 / 4 | Guo et al., ICML 2023 |
| PBM-PLDP | `PBM_PLDP` | ε-PLDP | 1 / 2 | Privatized Binomial Mechanism |
| RQM | `RQM` | ε-LDP | 1 / 2 | Youn et al. 2023 |
| CorQuant | `CorQuant` | None | 1 | Suresh et al. 2022 |
| SignSGD | `SignSGD` | None | 1 | Bernstein et al. 2019 |
| FedAvg | `FedAvg` | None | 32 | McMahan et al. 2016 |
| Gaussian LDP | `GaussianLDP` | ε-LDP | 32 | Balle & Wang 2018 |
| Gaussian CDP | `GaussianCDP` | ε-CDP | 32 | — |
| Laplace LDP | `LaplaceLDP` | ε-LDP | 32 | Dwork 2006 |

### Datasets

- **CIFAR-10** — IID and non-IID splits, ResNet18
- **MNIST** — IID and non-IID splits, CNN
- **FEMNIST** — IID and non-IID splits, CNN (LEAF benchmark)
- **Shakespeare** — IID and non-IID character-level splits, CharLSTM (LEAF benchmark)
- **Sent140** — sentiment classification, LSTM with pre-trained embeddings
- **Reddit** — next-word prediction, Transformer

## Requirements

- Python 3.11+
- PyTorch
- torchvision
- numpy
- pandas
- scipy
- pillow
- requests
- GPUtil
- ijson

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

### I-MVU Precomputation

The I-MVU method requires precomputed transition matrices. Run this once before using `IMVU`:

```bash
python precompute_imvu.py --epsilon 1 3 5 7 --budget 1 2 4
```

Parameters are saved to `imvu_params/imvu_eps{e}_b{b}.pt`. If a file is missing at runtime, the script will compute it on-the-fly (slower).

## Usage

```
python main.py --method {MethodName} --dataset {DatasetName} --device {CPU|GPU} [options]
```

`MethodName` in `{FedAvg, CorbinFL, AugCorbinFL, SignSGD, LDPFL, GaussianLDP, LaplaceLDP, GaussianCDP, CorQuant, IMVU, PBM_PLDP, RQM}`

`DatasetName` in `{CIFAR10, MNIST, FEMNIST, Shakespeare, Sent140, Reddit}`

### Command-line Arguments

**Core**

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | — | FL method (required) |
| `--dataset` | — | Dataset (required) |
| `--device` | — | `CPU` or `GPU` (required) |
| `--num_clients` | 50 | Number of FL clients |
| `--num_rounds` | 300 | Communication rounds |
| `--epsilon` | None | Privacy budget ε |
| `--lambda_param` | 1.0 | Global model smoothing parameter |
| `--dropout` | 0.0 | Client dropout probability |
| `--batch_size` | 64 | Local training batch size |
| `--seed` | 42 | Random seed |
| `--iid` | False | Use IID data split |
| `--eval_every` | 1 | Evaluate every N rounds |

**Optimizer**

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.001 | Learning rate (SignSGD step size; Adam lr) |
| `--weight_decay` | 0 | Weight decay |
| `--use_adam` | False | Use Adam-style server update |
| `--beta1` | 0.9 | Adam β₁ |
| `--beta2` | 0.999 | Adam β₂ |
| `--adam_eps` | 1e-8 | Adam ε |

**CorBin-FL / AugCorBin-FL**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_rand` | None | Number of shared random bits |
| `--Gamma` | 0.5 | Fraction of independent clients (AugCorBinFL) |

**I-MVU**

| Argument | Default | Description |
|----------|---------|-------------|
| `--imvu_budget` | 1 | Output bits per coordinate (1 / 2 / 4) |
| `--imvu_beta` | 1.0 | β scaling (1.0 for MNIST/CIFAR10; 32–128 for FEMNIST) |
| `--imvu_params_dir` | `imvu_params` | Directory with precomputed P matrices |

**PBM-PLDP**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pbm_m` | 1 | Bernoulli trials per coordinate (1=1-bit, 2=2-bit) |
| `--pbm_r_max` | None | Cap on encoding range (e.g. `1.0` for small ε) |

**RQM**

| Argument | Default | Description |
|----------|---------|-------------|
| `--rqm_m` | 2 | Quantization levels (2=1-bit, 3=2-bit) |
| `--rqm_q` | 0.5 | Interior level retention probability (only used when `rqm_m=3`) |
| `--rqm_r_max` | None | Cap on encoding range (e.g. `3.0` for small ε) |

**Resume**

| Argument | Description |
|----------|-------------|
| `--resume` | Resume from a previous checkpoint |
| `--checkpoint_path` | Path to `.pth` checkpoint file |
| `--results_path` | Path to results directory containing `config.txt` and CSV |

## Example Usage

### CorBin-FL

```bash
# CIFAR-10, epsilon=5
python main.py --method CorbinFL --dataset CIFAR10 --epsilon 5 --device GPU

# With dropout
python main.py --method CorbinFL --epsilon 10 --dropout 0.3 --device GPU

# MNIST with lambda tuning
python main.py --method CorbinFL --dataset MNIST --epsilon 10 --lambda_param 0.5 --device GPU
```

### Augmented CorBin-FL

```bash
python main.py --method AugCorbinFL --epsilon 10 --Gamma 0.2 --device GPU
```

### I-MVU

```bash
# 1-bit output (equivalent to LDPFL), epsilon=3
python main.py --method IMVU --dataset CIFAR10 --epsilon 3 --imvu_budget 1 --device GPU

# 2-bit output, lower variance at 2x communication cost
python main.py --method IMVU --dataset CIFAR10 --epsilon 3 --imvu_budget 2 --device GPU
```

### PBM-PLDP

```bash
# 1-bit, epsilon=5
python main.py --method PBM_PLDP --dataset CIFAR10 --epsilon 5 --pbm_m 1 --device GPU

# 2-bit, with range cap for small epsilon
python main.py --method PBM_PLDP --dataset CIFAR10 --epsilon 1 --pbm_m 1 --pbm_r_max 1.0 --device GPU
```

### RQM

```bash
# 1-bit (m=2)
python main.py --method RQM --dataset CIFAR10 --epsilon 5 --rqm_m 2 --device GPU

# 2-bit (m=3), interior level retention q=0.4
python main.py --method RQM --dataset CIFAR10 --epsilon 5 --rqm_m 3 --rqm_q 0.4 --device GPU
```

### SignSGD (non-private baseline)

```bash
python main.py --method SignSGD --dataset CIFAR10 --lr 0.0003 --device GPU
```

### FEMNIST

```bash
# non-IID split (default)
python main.py --method CorbinFL --dataset FEMNIST --epsilon 5 --device GPU

# IID split
python main.py --method CorbinFL --dataset FEMNIST --epsilon 5 --iid --device GPU
```

### Shakespeare

```bash
python main.py --method CorbinFL --dataset Shakespeare --epsilon 5 --device GPU
```

### Resuming a run

```bash
python main.py --resume \
    --checkpoint_path checkpoints/CorbinFL_CIFAR10_20240101_120000/model.pth \
    --results_path results/CorbinFL_CIFAR10_20240101_120000 \
    --device GPU
```

## Experiment Scripts

Pre-built experiment sweeps (epsilon × lambda × seed) are in the `Experiments/` directory:

```bash
cd Experiments
bash Cifar10_CorbinFL.sh
bash Cifar10_IMVU.sh
bash Cifar10_PBM_PLDP.sh
bash Cifar10_RQM.sh
bash Cifar10_SignSGD.sh
bash MNIST_CorbinFL.sh
bash FEMNIST_CorbinFL.sh
bash Shakespear_CorbinFL.sh
# ... and more
```

## Results

Results are saved as CSV files under `results/<experiment_name>/`, with columns:
`Round, Train Accuracy, Train Loss, Val Accuracy, Val Loss, Test Accuracy, Test Loss`.

## Checkpoints

Model checkpoints are saved to `checkpoints/<experiment_name>/model.pth` whenever validation accuracy improves. Use `--resume` to continue interrupted runs.

## Citation

If you use this work, please cite the paper.



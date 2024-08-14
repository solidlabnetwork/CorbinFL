# Federated Learning with Differential Privacy

This project implements CorBin-FL and Augmented CorBin-FL methods and compare them with other differentially private quantization method (LDP-FL Sun et. al. 2020) and non-private quantization method (CorQunt Sures et.al. 2022) and vanilla FL method. Also implement Improve Gaussian mechanism and Laplace mechanism for comparison. Implementation of federated learning is on CIFAR10 and MNIST datasets.

## Features

- Supports FL methods:
  - LDPFL (Local Differential Privacy Federated Learning "Sun, Qian, and Chen 2020")
  - CorBinFL (Correlated Binary Federated Learning "Ours")
  - CorQuant (Correlated Quantization "Suresh et al. 2022")
  - AugCorBinFL (Augmented Correlated Binary Federated Learning "Ours")
  - VanillaFL (Standard Federated Learning "FedAvg" "McMahan et al. 2016")
  - GaussianFL (Gaussian Noise Federated Learning "Balle and Wang 2018")
  - LaplaceFL (Laplace Noise Federated Learning "Dwork 2006")
- Dropout variants for most methods
- Supports MNIST and CIFAR10 datasets
- Implements ResNet18 and CNN architectures

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- pandas
- argparse

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script with desired arguments:

```
python main.py --method {MethodName} --dataset {DatasetName} --num_clients {int} --num_rounds {int} --epsilon {>0} --num_rand {int}
```
MethodName in {"LDPFL", "CorBinFL", "CorQuant", "AugCorBinFL", "VanillaFL", "GaussianFL", "LaplaceFL" }
DatasetName in {CIFAR10, MNIST}

### Command-line Arguments

- `--method`: FL method to use (default: "CorBinFL")
- `--dataset`: Dataset to use (default: "CIFAR10")
- `--device`: Device to use for computation (default: "GPU")
- `--num_clients`: Number of clients in FL (default: 50)
- `--num_rounds`: Number of communication rounds (default: 300)
- `--epsilon`: Privacy budget for Differential Privacy (default: 10)
- `--num_rand`: Number of random bits for CorBinFL (default: 5)
- `--gamma`: Gamma value for AugCorBinFL (default: 0)
- `--dropout`: Dropout probability (default: 0)
- `--lambda_param`: Smoothing parameter for global model update (default: 1)

## Project Structure

- `main.py`: Main script to run the FL experiments
- `Nets.py`: Contains neural network architectures (ResNet18, CNNMNIST)
- `main_dp_func.py`: Implements DP-related functions
- `main_utils.py`: Utility functions for data loading, GPU selection, etc.

## Results

Results will be saved in the `result` directory as Excel files, containing accuracy metrics and training time information.

## Checkpoints

Checkpoints will be saved in the `checkpoint` directory, allowing for resumption of training from the last saved state.

## Example Usage

To implement dropout in CorBin-FL, set the parameter dropout>0 for instance:
```
python main.py --method CorBinFL --epsilon 10 --dropout 0.3 
```

To implement the Augmented CorBin-FL, the parameter gamma needs to be set, for instance:
```
python main.py --method AugCorBinFL --epsilon 10 --gamma 0.2
```

To implement on MNIST dataset:
```
python main.py --method CorBinFL --dataset MNIST --epsilon 10 --lambda_param 0.5
python main.py --method CorBinFL --dataset MNIST --epsilon 10 --lambda_param 0.5
```

To implement dropout:
```
python main.py --method LDPFL --epsilon 5 --dropout 0.1 --lambda_param 0.4
```
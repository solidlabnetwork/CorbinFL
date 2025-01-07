# main.py
import os
import torch
import pandas as pd
from datetime import datetime
from typing import List, Tuple
import json

from main_dp_func import (
    load_data,
    save_checkpoint,
    load_checkpoint
)
from main_utils import (
    setup_device,
    set_seed,
)
from Nets import ResNet18, CNNMNIST, CharLSTM, Sent140LSTM
from methods.corbin_fl import CorBinFL
from methods.FedAvg import FedAvg
from federated_trainer import FederatedTrainer

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Federated Learning with CorBinFL")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['MNIST', 'CIFAR10', 'Shakespeare', 'Sent140'],
                        help='Dataset to use')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['FedAvg', 'CorBinFL'],
                        help='Federated learning method')
    parser.add_argument('--iid', action='store_true',
                        help='Whether to use IID data splitting (for Shakespeare)')
    parser.add_argument('--device', type=str, required=True,
                       choices=['CPU', 'GPU'],
                       help='Device to use for training')
    
    # Optional arguments with defaults
    parser.add_argument('--num_clients', type=int, default=50,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=300,
                       help='Number of communication rounds')
    parser.add_argument('--epsilon', type=float, default=10.0,
                       help='Privacy budget')
    parser.add_argument('--num_rand', type=int, default=5,
                       help='Number of random bits for CorBinFL')
    parser.add_argument('--lambda_param', type=float, default=1.0,
                       help='Aggregation parameter')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Client dropout probability')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N rounds')
    
    return parser.parse_args()

def setup_experiment_tracking(args) -> Tuple[str, str]:
    """Setup directories and files for experiment tracking"""
    # Create timestamp for unique experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.method}_{args.dataset}_{timestamp}"
    
    # Setup directories
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    results_dir = os.path.join('results', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create paths
    checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
    results_path = os.path.join(results_dir, 'metrics.csv')
    
    # Save experiment config
    config_path = os.path.join(results_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
            
    return checkpoint_path, results_path

def save_results(results: List[List[float]], results_path: str):
    """Save training results to CSV"""
    columns = [
        'Round', 
        'Train Accuracy', 'Train Loss',
        'Val Accuracy', 'Val Loss',
        'Test Accuracy', 'Test Loss'
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(results_path, index=False)

def get_model_creator(dataset_name: str):

    if dataset_name == "CIFAR10":
        lr = 0.001
        weight_decay = 0.003
    elif dataset_name == "MNIST":
        lr = 0.001
        weight_decay = 0.003
    elif dataset_name.lower() in ["shakespeare", "sent140"]:
        lr = 0.001
        weight_decay = 0

    def create_model(device):
        if dataset_name.lower() == "cifar10":
            model = ResNet18().to(device)
        elif dataset_name.lower() == "mnist":
            model = CNNMNIST().to(device)
        elif dataset_name.lower() == "shakespeare":
            model = CharLSTM().to(device)
        elif dataset_name.lower() == "sent140":
            vocab_file = 'data/Sent140/embs.json'
            with open(vocab_file, 'r') as f:
                data = json.load(f)
                vocab = data['vocab']  # List of words in the vocabulary
                vocab_size = len(vocab)

            # print(f"Vocab size: {vocab_size}")
            model = Sent140LSTM(vocab_size).to(device)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return model
    return create_model, lr, weight_decay

def main():
    # Parse arguments and setup
    args = parse_arguments()
    device = setup_device(args.device)
    set_seed(args.seed)
    
    # Setup experiment tracking
    checkpoint_path, results_path = setup_experiment_tracking(args)
    
    print("=== Experiment Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Privacy budget (ε): {args.epsilon}")
    print(f"Device: {args.device}")
    print(f"Checkpoint path: {checkpoint_path}")
    print("==============================\n")
    
    # Load data
    print("Loading data...")
    train_loaders, val_loader, test_loader, _ = load_data(
        args.dataset, 
        f'./data/{args.dataset}', 
        args.num_clients, 
        args.batch_size
    )
    
    # Initialize model
    print("Initializing model...")
    # model = ResNet18().to(device) if args.dataset == "CIFAR10" else CNNMNIST().to(device)
    model_creator, lr, weight_decay = get_model_creator(args.dataset)
    # print(f"GPU memory after model init: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Initialize method
    if args.method == "FedAvg":
        method = FedAvg(device=device)
    elif args.method == "CorBinFL":
        method = CorBinFL(
            epsilon=args.epsilon,
            num_rand=args.num_rand,
            lambda_param=args.lambda_param,
            dropout=args.dropout,
            device=device
        )
    
    # Initialize trainer
    trainer = FederatedTrainer(
            method=method,
            model=model_creator(device),  # Initial model
            model_creator=model_creator,  # Function to create new models
            dataset_name=args.dataset, 
            train_loaders=train_loaders,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_rounds=args.num_rounds,
            checkpoint_path=checkpoint_path,
            eval_every=args.eval_every,
            lr = lr,
            weight_decay = weight_decay
        )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    counter = 0
    results = []
    
    for round in range(args.num_rounds):
        # Train round and get accuracies
        accuracies = trainer.train_round(round)
        
        if accuracies is not None:  # Only when evaluation is performed
            (train_acc, train_loss), (val_acc, val_loss), (test_acc, test_loss) = accuracies
            results.append([
                round + 1,
                train_acc, train_loss,
                val_acc, val_loss,
                test_acc, test_loss
            ])
            
            print(f'\nRound {round + 1}:')
            print(f'Train Accuracy: {train_acc:.2f}%')
            print(f'Val Accuracy: {val_acc:.2f}%')
            print(f'Test Accuracy: {test_acc:.2f}%')
            print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
            
            # Save checkpoint if improved
            if val_acc > best_val_acc:
                counter = 0
                best_val_acc = val_acc
                save_checkpoint(
                    checkpoint_path,
                    round,
                    trainer.model,
                    [test_acc],
                    [train_acc],
                    results,
                    counter,
                    best_val_acc
                )
                print(f"New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                counter += 1
            
            # Early stopping check
            if counter >= 5:
                counter = 0
                print("Loading best model due to no improvement")
                torch.cuda.empty_cache()
                _, _, _, _, _, _ = load_checkpoint(checkpoint_path, trainer.model)
            
            # Save current results
            save_results(results, results_path)
        
        # Memory stats every 10 rounds
        if round % 10 == 0:
            print(f"\nMax GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            torch.cuda.reset_peak_memory_stats()

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
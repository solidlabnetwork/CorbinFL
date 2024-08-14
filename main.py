import time
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn


from Nets import ResNet18, CNNMNIST
from main_dp_func import (
    perturb_weight,
    comm_rand,
    one_dim_one_bit_cq,
    compute_center_and_range,
    federated_learning_pairing,
    train_on_client,
    client_assignment,
    compute_sigma_agm,
    load_checkpoint,
    save_checkpoint,
    load_data
)
from main_utils import (
    setup_device,
    set_seed,
    validate,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for your model")
    parser.add_argument('--method', type=str, default="CorBinFL", 
                        choices=["LDPFL", "CorBinFL", "CorQuant", "AugCorBinFL", "VanillaFL", "GaussianFL", "LaplaceFL"],
                        help='Method for Federated Learning')
    parser.add_argument('--dataset', type=str, default="CIFAR10", choices=["MNIST", "CIFAR10"],
                        help='Dataset to use')
    parser.add_argument('--device', type=str, default="GPU", choices=["GPU", "CPU"],
                        help='Device that is used for Federated Learning')
    parser.add_argument('--num_clients', type=int, default=50,
                        help='Number of Clients in Federated Learning')
    parser.add_argument('--num_rounds', type=int, default=300,
                        help='Round of communication in Federated Learning')
    parser.add_argument('--epsilon', type=float, default=10,
                        help='Privacy budget for Differential Privacy')
    parser.add_argument('--num_rand', type=int, default=5,
                        help='Number of Randomness for CorBinFL')
    parser.add_argument('--gamma', type=float, default=0,
                        help='Gamma value for AugCorBinFL')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout probability for CorBinDropout')
    parser.add_argument('--lambda_param', type=float, default=1,
                        help='Smoothing parameter for the global model update')
    return parser.parse_args()




def main():
    args = parse_arguments()
    device = setup_device(args.device)
    set_seed(42)

    # Setup hyperparameters
    n_clients = args.num_clients
    num_rounds = args.num_rounds
    epsilon = torch.tensor(args.epsilon, device=device)
    alpha = (torch.exp(epsilon) + 1) / (torch.exp(epsilon) - 1)

     # raise error if dropout is greater than 1 or less than 0
    if args.dropout > 1 or args.dropout < 0:
        raise ValueError("Dropout probability should be between 0 and 1")
    dropout = torch.tensor(args.dropout, dtype=torch.float32, device=device)

    if args.method not in ["AugCorBinFL"]: # we didnt implement dropout for AugCorBinFL
        method = args.method + "Dropout" if dropout > 0 else args.method
    else:
        method = args.method
    lambda_param = args.lambda_param
    NumRand = args.num_rand if "CorBin" in method else 0
    Gamma = args.gamma if "AugCorBinFL" in method else 0

    delta = 1e-5 # delta for differential privacy of Gaussian method
    n_local_epochs = 1
    learning_rate = 0.001
    weight_decay = 0.003
    counter_reset = 5

    print('Method:', method)

    # Load data
    print(args.dataset)
    data_dir = f'./data/{args.dataset}'
    os.makedirs(data_dir, exist_ok=True)
    client_dataloaders, val_loader, test_dataloader, _ = load_data(args.dataset, data_dir, n_clients)
    global_model = ResNet18().to(device) if args.dataset == "CIFAR10" else CNNMNIST().to(device)

    
    # Setup model and criteria
    criteria = nn.CrossEntropyLoss()
    
    # Setup checkpoint
        # Generate the Name for checkpoint and results
    if method == "CorBinFL":
        Name = f'{method}_{args.dataset}_eps_{args.epsilon}_NR_{args.num_rand}_lmbda_{lambda_param}_num_clients_{n_clients}_epc{num_rounds}'   
    elif method == "AugCorBinFL":
        Name = f'{method}_{args.dataset}_eps_{args.epsilon}_NR_{args.num_rand}_lmbda_{lambda_param}_Gamma_{args.gamma}_num_clients_{n_clients}_epc{num_rounds}'
    elif method == "CorBinFLDropout":
        Name = f'{method}_{args.dataset}_eps_{args.epsilon}_NR_{args.num_rand}_lmbda_{lambda_param}_dropout_{args.dropout}_num_clients_{n_clients}_epc{num_rounds}'
    elif method == "CorQuant":
        Name = f'{method}_{args.dataset}_lmbda_{lambda_param}_num_clients_{n_clients}_epc{num_rounds}'
    elif method == "CorQuantDropout":
        Name = f'{method}_{args.dataset}_lmbda_{lambda_param}_dropout_{args.dropout}_num_clients_{n_clients}_epc{num_rounds}'
    elif method == "VanillaFL":
        Name = f'{method}_{args.dataset}_num_clients_{n_clients}_epc{num_rounds}'
    elif method == "LDPFLDropout":
        Name = f'{method}_{args.dataset}_eps_{args.epsilon}_lmbda_{lambda_param}_dropout_{args.dropout}_num_clients_{n_clients}_epc{num_rounds}'
    else:
        Name = f'{method}_{args.dataset}_eps_{args.epsilon}_lmbda_{lambda_param}_num_clients_{n_clients}_epc{num_rounds}'

    print("*" * 50)
    print("Dataset:", args.dataset)
    print("Method:", Name)
    print("*" * 50)
    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{Name}.pth')

    start_round, accuracy_list, train_accuracy_list, results, counter, best_accuracy = load_checkpoint(checkpoint_path, global_model)

    print(f"Starting from round {start_round + 1}")
    start_time = time.time()
 
    for round in range(start_round, num_rounds):
        round_start_time = time.time()

        
        # Compute center and range for each layer
        C, R = compute_center_and_range(global_model, device)
        global_state_dict = global_model.state_dict()
        c_r_keys = [k for k in global_state_dict.keys() if "weight" in k or "bias" in k]
        c_r_mapping = {k: i for i, k in enumerate(c_r_keys)}
        if method in ["CorBinFL", "AugCorBinFL", "CorBinFLDropout"]:
            client_pairing = federated_learning_pairing(n_clients)
        else:
            client_pairing = [i for i in range(n_clients)]
        client_assignment_r = client_assignment(n_clients, method, gamma=Gamma, dropout=dropout, device=device)

        if "CorQuant" in method:
            pi_dict = {}
            for k in c_r_keys:
                shape = global_state_dict[k].shape
                pi_dict[k] = torch.stack([torch.randperm(n_clients, device=device) for _ in range(torch.prod(torch.tensor(shape)))]).reshape(*shape, n_clients)

        # Local training
        local_models = []
        for i in range(n_clients):
            
            client_model = ResNet18().to(device) if args.dataset == "CIFAR10" else CNNMNIST().to(device)
            client_model.load_state_dict(global_state_dict)
            client_model = train_on_client(client_model, client_dataloaders[client_pairing[i]], epochs=n_local_epochs, lr=learning_rate, weight_decay=weight_decay, device=device)
            client_state_dict = client_model.state_dict()

            if method in ["CorBinFL", "AugCorBinFL"]:
                if client_assignment_r[i] == 1:  # if the client is leader
                    total_params = sum(client_state_dict[key].numel() for key in c_r_keys)
                    CR = comm_rand(args.num_rand, total_params, device=device)
                    UP = 1
                elif client_assignment_r[i] == 0:  # if the client is follower
                    UP = -1
            elif method == "CorBinFLDropout":
                if client_assignment_r[i] == 1:  # if the client is leader
                    total_params = sum(client_state_dict[key].numel() for key in c_r_keys)
                    CR = comm_rand(args.num_rand, total_params, device=device)
                    UP = 1
                elif client_assignment_r[i] == 0:  # if the client is follower
                    UP = -1
                    if client_assignment_r[i-1] == 2:  # if the client is follower and the leader dropout create a new CR
                        total_params = sum(client_state_dict[key].numel() for key in c_r_keys)
                        CR = comm_rand(args.num_rand, total_params, device=device)

            start_idx = 0
            for key in c_r_keys:
                param_ = client_state_dict[key]
                param_data_ = param_.data
                num_elements = param_data_.numel()
                c = C[c_r_mapping[key]].item()
                r = R[c_r_mapping[key]].item()



                if method in ["LDPFL", "LDPFLDropout"]:
                    if client_assignment_r[i] < 2:
                        client_state_dict[key] = perturb_weight(param_data_, alpha, c, r, LDPFL=True, device=device)
                elif method in ["CorBinFL", "CorBinFLDropout"]:
                    if client_assignment_r[i] < 2: # if the client is not in dropout (For CorBinFL client_assignment_r[i] is always <2)
                        CR_layer = CR[start_idx:start_idx + num_elements]
                        client_state_dict[key]= perturb_weight(param_data_, alpha, c, r, CR_layer, NumRand, UP, LDPFL=False, device=device)
                elif method == "AugCorBinFL": # We didn't implement dropout for AugCorBinFL
                    if client_assignment_r[i] == 2: # if the client is assigned to be LDPFL
                        client_state_dict[key] = perturb_weight(param_data_, alpha, c, r, LDPFL=True, device=device)
                    else:
                        CR_layer = CR[start_idx:start_idx + num_elements]
                        client_state_dict[key]= perturb_weight(param_data_, alpha, c, r, CR_layer, NumRand, UP, LDPFL=False, device=device)
                elif method in ["CorQuant", "CorQuantDropout"]:
                    if client_assignment_r[i] < 2: # if the client is not in dropout
                        pi = pi_dict[key][..., i].to(device)
                        client_state_dict[key] = one_dim_one_bit_cq(param_data_,  pi, c, r, n_clients, device=device)
                elif method in ["GaussianFL", "GaussianFLDropout"]:
                    if client_assignment_r[i] < 2:
                        sensitivity = 2 * r
                        sigma = compute_sigma_agm(epsilon, delta, sensitivity)
                        noise = torch.normal(0, sigma, size=param_data_.shape, device=device)
                        # noise = torch.distributions.Normal(loc=0, scale=sigma).sample(param_data_.shape).to(device)
                        client_state_dict[key].add_(noise)
                elif method in ["LaplaceFL", "LaplaceFLDropout"]:
                    if client_assignment_r[i] < 2:
                        sensitivity = 2 * r
                        scale = sensitivity / epsilon

                        noise = torch.distributions.Laplace(loc=0, scale=scale+1e-6).sample(param_data_.shape).to(device) # 1e-6 is added to avoid zero probability
                        client_state_dict[key].add_(noise)
                elif method != "VanillaFL":
                    raise ValueError(f"Unsupported method: {method}")


                start_idx += num_elements

            if "Dropout" in method:
                if client_assignment_r[i] < 2:  # do not consider the dropout clients
                    local_models.append(client_state_dict)
            else:
                local_models.append(client_state_dict)

        # Update global model
        for key in global_state_dict.keys():
            if global_state_dict[key].dtype == torch.long:
                mean_weights = torch.stack([local_model[key] for local_model in local_models], 0).float().mean(0).long()
            else:
                mean_weights = torch.stack([local_model[key] for local_model in local_models], 0).mean(0)
            
            global_state_dict[key] = (1 - lambda_param) * global_state_dict[key] + lambda_param * mean_weights

        global_model.load_state_dict(global_state_dict)

        # Evaluation
        global_model.eval()
        _, test_accuracy = validate(global_model, test_dataloader, criteria, device=device)
        _, val_accuracy = validate(global_model, val_loader, criteria, device=device)
        _, train_accuracy = validate(global_model, client_dataloaders[0], criteria, device=device)

        accuracy_list.append(test_accuracy)
        print(f'Round {round + 1}, Test Accuracy: {test_accuracy:.2f}%')
        print(f'Round {round + 1}, Train Accuracy: {train_accuracy:.2f}%')
        results.append([round + 1, train_accuracy, test_accuracy])

        # Save best model
        if val_accuracy > best_accuracy:
            counter = 0
            best_accuracy = val_accuracy
            print(f"Saving the best model with test accuracy: {test_accuracy:.2f}%")
            save_checkpoint(checkpoint_path, round, global_model, accuracy_list, train_accuracy_list, results, counter, best_accuracy)
        else:
            counter += 1
            print("Counter:", counter)

        if counter >= counter_reset:
            counter = 0
            print("Reloading the best model and continuing training.")
            if os.path.exists(checkpoint_path):
                print("Loading the best model from the checkpoint.")
                checkpoint = torch.load(checkpoint_path)
                global_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("No checkpoint found. Exiting the training loop.")
                break
        print("Best validation accuracy so far:", best_accuracy)

        round_end_time = time.time()
        print(f"Round Time: {round_end_time - round_start_time:.2f} seconds")

    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")
    
    # Save results as csv
    results_df = pd.DataFrame(results, columns=["Round", "Train Accuracy", "Test Accuracy"])
    results_df["Total Training Time"] = end_time - start_time

    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{Name}.csv"
    csv_path = os.path.join(output_dir, filename)
    results_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
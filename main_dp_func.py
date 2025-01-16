# main_dp_func.py
import torch
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import os
from typing import List
import random
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy import optimize
from dataloaders.mnist import MNISTDataLoader
from dataloaders.cifar10 import CIFAR10DataLoader
from dataloaders.shakespeare_IID import ShakespeareDataLoader
from dataloaders.sentiment140 import Sent140DataLoader
from dataloaders.reddit import RedditDataLoader


def train_on_client(client_model, dataloader, epochs=1, lr=0.001, weight_decay=0.003, 
                   device='cpu', compute_gradients=False, 
                   momentum_state=None, beta=0.9, dataset_name='mnist'): 
    """
    Train a client model and return either the updated model or the accumulated gradients
    
    Args:
        compute_gradients (bool): If True, return gradients instead of updated model
    Returns:
        If compute_gradients=True: Dictionary of accumulated gradients
        If compute_gradients=False: Updated model
    
    Train client model and handle momentum at client level
    
    Args:
        momentum_state: Current momentum state for this client
        beta: Momentum constant β from algorithm
    """
    optimizer = torch.optim.Adam(client_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Check if it's Reddit dataset
    is_reddit = dataset_name.lower() == 'reddit'
    criterion = (nn.CrossEntropyLoss(ignore_index=dataloader.dataset.pad_idx) 
                if is_reddit else nn.CrossEntropyLoss())
    
    client_model.train()
    
    # Initialize gradient accumulation if needed
    if compute_gradients:
        # Initialize gradient accumulation
        gradient_dict = {}
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                gradient_dict[name] = torch.zeros_like(param.data)
                
        # Initialize momentum if not provided
        if momentum_state is None:
            momentum_state = {}
            for name, param in client_model.named_parameters():
                if param.requires_grad:
                    momentum_state[name] = torch.zeros_like(param.data)
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Handle different dataset formats
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients for this batch
            optimizer.zero_grad()
            
            # Forward pass
            if is_reddit:
                padding_mask = (inputs == dataloader.dataset.pad_idx)
                outputs = client_model(inputs, src_padding_mask=padding_mask)
                outputs = outputs.contiguous().view(-1, outputs.size(-1))
                targets = targets.contiguous().view(-1)
            else:
                outputs = client_model(inputs)
            
            # Compute loss and backpropagate
            loss = criterion(outputs, targets)
            loss.backward()
            
            if compute_gradients:
                # Accumulate gradients
                for name, param in client_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        gradient_dict[name] += param.grad.data
            else:
                optimizer.step()
    
    if compute_gradients:
        # Average gradients over batches
        num_batches = len(dataloader)
        for name in gradient_dict:
            gradient_dict[name] /= num_batches
            
            # Update momentum according to algorithm:
            # vm ← (1-β)g̃m + βvm
            momentum_state[name] = ((1 - beta) * gradient_dict[name] + 
                                  beta * momentum_state[name])
            
            # Return sign of momentum
            gradient_dict[name] = torch.sign(momentum_state[name])
            
        return gradient_dict, momentum_state  # Return both updated gradient signs and momentum
    else:
        return client_model, None # Return updated model


def compute_center_and_range(model, device='cpu'):
    C, R = [], []
    for param in model.parameters():
        param_data = param.data
        range_ = 0.5 * (torch.max(param_data) - torch.min(param_data))
        center = (torch.max(param_data)+torch.min(param_data))/2
        C.append(center)
        R.append(range_)
    R = torch.tensor(R, dtype=torch.float32).to(device)
    C = torch.tensor(C, dtype=torch.float32).to(device)

    return C, R

def federated_learning_pairing(num_clients):

    clients = list(range(num_clients))
    random.shuffle(clients)    
    num_pairs = num_clients // 2
    # Create pairs, leaving out the last client if odd
    pairs = [(clients[i], clients[i+1]) for i in range(0, num_pairs * 2, 2)]
    
    output = clients.copy()
    
    # For each pair, find the leader and bring the leader to position before the follower
    for i in range(len(pairs)):
        Y_i = np.random.binomial(1, 0.5)
        Y_j = np.random.binomial(1, 0.5)
        
        if Y_i != Y_j: # if Y_i = Y_j then the leader is already in the correct position
            idx_i = output.index(pairs[i][0])
            idx_j = output.index(pairs[i][1])
            output[idx_i], output[idx_j] = output[idx_j], output[idx_i]
    
    # output = range(num_clients)
    return output # returen a vector [leader1, follower1, leader2, follower2, ...]


def find_index_batch(NumRand, P_batch):
    """
    Find indices for a batch of probabilities in a list with 2^d elements from 0 to 1.
    
    Parameters:
    P_batch (torch.Tensor): Batch of probabilities
    d (int): Determines the number of elements in ProbList (2^d)
    
    Returns:
    torch.Tensor: Indices for each probability in the batch
    """
    # Calculate 2^d
    two_pow_d = 2 ** NumRand
    
    # Multiply P_batch by 2^d and floor it to get the indices
    indices = torch.floor(P_batch * two_pow_d).long()
    
    # Clamp the indices to be within [0, 2^d - 1]
    indices = torch.clamp(indices, 0, two_pow_d - 1)

    return indices.to(P_batch.device)

def comm_rand(NumRand, NumParam, device='cpu'):
    return torch.randint(0, 2 ** NumRand, (NumParam,), device=device)


def perturb_weight(W,  alpha, c, r, CR=None, NumRand=None, UP=None, LDPFL=True, device='cpu'):

    W = W.to(device)
    shape = W.shape
    W= W.view(-1)

    # Calculate ProbMarginal
    ProbMarginal = 0.5 + (torch.clamp(W, c - r, c + r) - c) / (2 * alpha * r)

    if LDPFL:
        # Create U tensor for LDPFL case
        U = torch.where(torch.rand_like(W) < ProbMarginal, 1.0, -1.0)
    else:
        CR = CR.to(device)
        # Calculate P based on UP
        P = ProbMarginal if UP == 1 else 1 - ProbMarginal

        idx = find_index_batch(NumRand, P)
        IntCR = CR.long()
         # Create initial U tensor
        U = torch.full_like(W, UP, dtype=torch.float32)
        U = torch.where(IntCR >= idx + 1, -UP, U)


        middle_case = (IntCR >= idx) & (IntCR < idx + 1)

        # Handle edge cases
        idx = torch.where(idx + 1 >= 2**NumRand, 2**NumRand - 2, idx)

        if middle_case.any():
            # Calculate probabilities for idx 
            prob_idx = idx.float() / 2**NumRand

            # Calculate ProbGrid for middle case
            ProbGrid = torch.zeros_like(W)
            ProbGrid[middle_case] = (P[middle_case] - prob_idx[middle_case]) * 2**NumRand

            # Generate random values and apply to middle case
            random_values = torch.rand_like(W)
            UP_tensor = torch.tensor(UP, dtype=torch.float32, device=device)
            U[middle_case] = torch.where(random_values[middle_case] < ProbGrid[middle_case],
                                         UP_tensor,
                                         -UP_tensor)

    # Update W for all elements at once
    W = c + U * alpha * r

    return W.view(shape)



def one_dim_one_bit_cq(W, pi, c, r, n_clients, device='cpu'):
    """
    Implement the quantization process for a tensor as per the OneDimOneBitCQ algorithm.
    
    Parameters:
    W (torch.Tensor): The input tensor to be quantized (weights)
    pi (torch.Tensor): The permutation tensor for the current client, same shape as W
    c (float): Center of the range
    r (float): Half-width of the range
    n_clients (int): Number of clients
    device (torch.device): The device to perform computations on
    
    Returns:
    torch.Tensor: The quantized tensor Q(W) with the same shape as W
    """
    # Move input to the specified device
    W = W.to(device)
    pi = pi.to(device)
    
    # Calculate range bounds
    l = c - r
    u = c + r
    
    # Clamp W in range [c-r, c+r]
    W = torch.clamp(W, min=l, max=u)
    
    # Step 1: Calculate y
    y = (W - l) / (2 * r)
    
    # Step 2: Calculate U
    gamma = torch.rand_like(W, device=device) / n_clients
    U = (pi.float() / n_clients) + gamma
    
    # Step 3: Determine Q(W)
    Q = 2 * r * (U < y).float() + l
    
    return Q



def client_assignment(num_clients, method, gamma=0, dropout=0, device='cpu'):
    if method == 'CorBinFL':
        return torch.tensor([1 if i % 2 == 0 else 0 for i in range(num_clients)], device=device)
    
    elif method == 'AugCorBinFL':

        assignments = torch.full((num_clients,), -1, dtype=torch.long, device=device)     
        # Calculate number of clients that will do LDPFL
        LDPFL_clients = int(num_clients * gamma)
        # Assign 2 to Randomly selected LDPFL clients (2)
        LDPFL_indices = torch.randperm(num_clients, device=device)[:LDPFL_clients]
        assignments[LDPFL_indices] = 2
        # Assign leaders (1) and followers (0) to remaining positions
        empty_positions = torch.nonzero(assignments == -1).squeeze()
        for i in range(0, len(empty_positions), 2):
            if i < len(empty_positions):
                assignments[empty_positions[i]] = 1  # Leader
            if i + 1 < len(empty_positions):
                assignments[empty_positions[i + 1]] = 0  # Follower       
        # If there's an odd number of empty positions, make the last one a leader
        if len(empty_positions) % 2 != 0:
            assignments[empty_positions[-1]] = 1
        
        return assignments
    elif 'Dropout' in method:
        assignments = torch.full((num_clients,), -1, dtype=torch.long, device=device) 
        # Use biased coin flip to determine non-contributing clients (2)
        coin_flips = torch.bernoulli(torch.full((num_clients,), dropout, device=device))
        assignments[coin_flips == 1] = 2
        
        # Find empty positions
        empty_positions = torch.nonzero(assignments == -1).squeeze()
        
        # Assign leaders (1) and followers (0) to remaining positions
        for i in range(0, len(empty_positions), 2):
            if i < len(empty_positions):
                assignments[empty_positions[i]] = 1  # Leader
            if i + 1 < len(empty_positions):
                assignments[empty_positions[i + 1]] = 0  # Follower
        
        # If there's an odd number of empty positions, make the last one a leader
        if len(empty_positions) % 2 != 0:
            assignments[empty_positions[-1]] = 1
        
        return assignments
    else:
        return torch.zeros(num_clients, device=device)
    

def compute_sigma_agm(epsilon, delta, sensitivity):
    """
    Compute the optimal sigma for the Gaussian mechanism using the Analytical Gaussian Mechanism (AGM).

    Balle B, Wang YX. 
    Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. 
    In International Conference on Machine Learning 2018 Jul 3 (pp. 394-403). PMLR.

    Inputs:
    - epsilon (float): The epsilon parameter for differential privacy.
    - delta (float): The delta parameter for differential privacy.
    - sensitivity (float): The sensitivity of the query.

    Output:
    - sigma (float): The optimal standard deviation for the Gaussian mechanism.

    The function computes the tightest possible sigma for the Gaussian mechanism
    that satisfies (epsilon, delta)-differential privacy for a given query sensitivity.
    """
    def phi(t):
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    def B_plus(v):
        return phi(math.sqrt(epsilon * v)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (v + 2)))

    def B_minus(u):
        return phi(-math.sqrt(epsilon * u)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (u + 2)))

    delta_0 = phi(0) - math.exp(epsilon) * phi(-math.sqrt(2 * epsilon))

    if delta >= delta_0:
        v_star = optimize.brentq(lambda v: B_plus(v) - delta, 0, 100)
        alpha = math.sqrt(1 + v_star / 2) - math.sqrt(v_star / 2)
    else:
        u_star = optimize.brentq(lambda u: B_minus(u) - delta, 0, 100)
        alpha = math.sqrt(1 + u_star / 2) + math.sqrt(u_star / 2)

    return alpha * sensitivity / math.sqrt(2 * epsilon)



def load_data(dataset_name: str, data_dir: str, n_clients: int, batch_size=64, iid=True):
    """Load and prepare dataset for federated learning
    
    Args:
        dataset_name (str): Name of the dataset ('mnist', 'cifar10', or 'shakespeare')
        data_dir (str): Directory path for dataset
        n_clients (int): Number of clients for federated learning
        batch_size (int): Batch size for dataloaders
        iid (bool): Whether to use IID data distribution (for Shakespeare dataset)
    
    Returns:
        tuple: (client_dataloaders, val_loader, test_dataloader, input_channels)
    """
    dataset_loaders = {
        'mnist': MNISTDataLoader,
        'cifar10': CIFAR10DataLoader,
        'shakespeare': ShakespeareDataLoader,
        'sent140': Sent140DataLoader,
        'reddit': RedditDataLoader
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Initialize the appropriate dataloader
    dataloader = dataset_loaders[dataset_name](
        data_dir=data_dir,
        n_clients=n_clients,
        batch_size=batch_size,
        iid=iid
    )
    
    # Load and prepare the data
    return dataloader.load_data()

# def distribute_data_iid(dataset, num_users):
#     num_items = int(len(dataset) / num_users)
#     all_idxs = np.arange(len(dataset))
#     np.random.shuffle(all_idxs)
#     dict_users = {i: set(all_idxs[i * num_items:(i + 1) * num_items]) for i in range(num_users)}
#     return dict_users

# class DatasetSubset(Dataset):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = list(indices)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         data_idx = self.indices[idx]
#         return self.dataset[data_idx]

def load_checkpoint(checkpoint_path: str, global_model: nn.Module):
    if os.path.exists(checkpoint_path):
        print("Loading the checkpoint from", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        global_model.load_state_dict(checkpoint['model_state_dict'])
        start_round = checkpoint['round']
        accuracy_list = checkpoint['accuracy_list']
        train_accuracy_list = checkpoint['train_accuracy_list']
        results = checkpoint['results']
        counter = checkpoint['counter']
        best_accuracy = checkpoint['best_val_accuracy']
        
        # # Restore the saved states if they exist
        # for state_name in ['torch_rng_state', 'cuda_rng_state', 'numpy_rng_state', 'python_rng_state']:
        #     if state_name in checkpoint:
        #         if state_name == 'torch_rng_state':
        #             torch.set_rng_state(checkpoint[state_name])
        #         elif state_name == 'cuda_rng_state':
        #             torch.cuda.set_rng_state_all(checkpoint[state_name])
        #         elif state_name == 'numpy_rng_state':
        #             np.random.set_state(checkpoint[state_name])
        #         elif state_name == 'python_rng_state':
        #             random.setstate(checkpoint[state_name])
        #     else:
        #         print(f"No {state_name} found in the checkpoint.")
        
        return start_round, accuracy_list, train_accuracy_list, results, counter, best_accuracy
    
    return 0, [], [], [], 0, 0.0

def save_checkpoint(checkpoint_path: str, round: int, global_model: nn.Module, 
                    accuracy_list: List[float], train_accuracy_list: List[float], 
                    results: List[List[float]], counter: int, best_accuracy: float):
    torch.save({
        'round': round + 1,
        'model_state_dict': global_model.state_dict(),
        'accuracy_list': accuracy_list,
        'train_accuracy_list': train_accuracy_list,
        'results': results,
        'counter': counter,
        'best_val_accuracy': best_accuracy,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate()
    }, checkpoint_path)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
import time
from methods.signsgd import SignSGD
import os

from main_dp_func import (
    compute_center_and_range,
    train_on_client,
    save_checkpoint,
    load_checkpoint
)

class FederatedTrainer:
    def __init__(
        self, 
        method: object,
        model: nn.Module,
        model_creator: object,
        train_loaders: List[DataLoader],
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        num_rounds: int,
        checkpoint_path: str,
        dataset_name: str,
        eval_every: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        data_loader: object = None,
        is_save: bool = False,
    ):
        self.method = method
        self.model = model
        self.model_creator = model_creator
        self.train_loaders = train_loaders
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset_name = dataset_name.lower()
        self.device = device
        self.num_rounds = num_rounds
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.eval_every = eval_every
        self.lr = lr
        self.weight_decay = weight_decay
        self.data_loader = data_loader
        self.client_momentums = {} # Store client momentum states
        self.is_save = is_save # Save C and R , and weightsif True
        
        if self.dataset_name == 'reddit':
            self.criteria = nn.CrossEntropyLoss(ignore_index=val_loader.dataset.pad_idx)
        else:
            self.criteria = nn.CrossEntropyLoss()

    def train_round(self, round_num: int) -> Optional[Tuple[float, float, float]]:
        round_start_time = time.time() 
        torch.cuda.empty_cache()
        
    
        
        # Compute center and range for each layer
        C, R = compute_center_and_range(self.model, self.device)

        # Save C and R if is_save is True
        if self.is_save:
            # os.makedirs(self.checkpoint_path, exist_ok=True)
            torch.save((C, R), os.path.join(self.checkpoint_dir, f'round_{round_num}_CR.pt'))

        global_state_dict = self.model.state_dict()
        
        # Initialize weighted sum dictionary with zeros
        weighted_sum = {}
        for key in global_state_dict:
            weighted_sum[key] = torch.zeros_like(global_state_dict[key],dtype=torch.float32, device=self.device)
        
        # For FEMNIST and shakespeare nonIID, get new clients each round
        if self.dataset_name.lower() in ['femnist', 'shakespeare'] and self.data_loader is not None:
            self.train_loaders, _, _, self.client_weights, _ = self.data_loader.load_data()
        
        # Initialize round-specific data
        n_clients = len(self.train_loaders) if self.data_loader is None else self.data_loader.n_clients
        round_data = self.method.initialize_round(n_clients, self.model)
        client_ordering = round_data['client_ordering']

        # check if the method is gradient based
        is_gradient_based = isinstance(self.method, (SignSGD))


        
        # Create mapping from parameter index to state_dict key
        param_to_key = {}
        idx = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'bias' in name:
                param_to_key[idx] = name
                idx += 1
        
        # Initialize lists to store updates if saving is enabled
        if self.is_save:
            updates_before = []
            updates_after = []
        
        
        # Local training - process clients in batches
        client_batch_size = 50
        # print("Length of train loader is", len(self.train_loaders))
        # print("client ordering is", client_ordering)
        for client_batch_start in range(0, n_clients, client_batch_size):
            client_batch_end = min(client_batch_start + client_batch_size, n_clients)
            
            for i in range(client_batch_start, client_batch_end):
                client_model = self.model_creator(self.device)
                client_model.load_state_dict(global_state_dict)
                actual_client_idx = client_ordering[i]
                # Get client's momentum state
                momentum_state = self.client_momentums.get(actual_client_idx)

                # Get appropriate loader based on dataset type
                if self.dataset_name == 'reddit':
                    client_loader = self.data_loader.get_client_dataloader(actual_client_idx)
                else:
                    client_loader = self.train_loaders[actual_client_idx]

                # Train on client
                client_update, new_momentum = train_on_client(
                    client_model, 
                    client_loader,
                    device=self.device,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    compute_gradients=is_gradient_based,
                    momentum_state=momentum_state,
                    beta=self.method.beta if isinstance(self.method, (SignSGD)) else None,
                    dataset_name=self.dataset_name
                      # Get beta from SignSGD method
                )
                # Store updated momentum
                if is_gradient_based:
                    self.client_momentums[actual_client_idx] = new_momentum
                
                # Store update before processing
                if self.is_save:
                    update_before = client_update if is_gradient_based else client_model.state_dict()
                    updates_before.append(update_before)

                # Process client update
                processed_state_dict = self.method.process_client_update(
                    client_update if is_gradient_based else client_model.state_dict(),
                    i,
                    {param_to_key[idx]: c for idx, c in enumerate(C)},
                    {param_to_key[idx]: r for idx, r in enumerate(R)},
                    round_data
                )
                
                # Store update after processing
                if self.is_save and processed_state_dict is not None:
                    updates_after.append(processed_state_dict)

                if processed_state_dict is not None:
                    # Get client weight
                    if hasattr(self, 'client_weights') and is_gradient_based==False:
                        weight = self.client_weights[actual_client_idx]
                    elif is_gradient_based:
                        weight = 1.0
                    else:
                        weight = 1.0 / n_clients
                    
                    # Add to weighted sum
                    for key in weighted_sum:
                        weighted_sum[key] += weight * processed_state_dict[key].float()
                
                del client_model
                if self.dataset_name == 'reddit':
                    del client_loader
                # torch.cuda.empty_cache()
            
            torch.cuda.empty_cache()
        
        # Save all updates at once if is_save is True
        if self.is_save:
            torch.save(updates_before, os.path.join(self.checkpoint_dir, f'round_{round_num}_updates_before.pt'))
            torch.save(updates_after, os.path.join(self.checkpoint_dir, f'round_{round_num}_updates_after.pt'))
        
        # Use method's aggregation logic for final update
        dummy_local_models = [weighted_sum]
        dummy_weights = [1.0]
        global_state_dict = self.method.aggregate_updates(
            global_state_dict, 
            dummy_local_models,
            dummy_weights
        )
        
        self.model.load_state_dict(global_state_dict)
        
        round_end_time = time.time()
        round_time = round_end_time - round_start_time
        print(f"Round {round_num + 1} completed in {round_time:.2f} seconds")
        
        # Only evaluate periodically
        if round_num % self.eval_every == 0:
            return self.evaluate()
        return None

    @torch.no_grad()
    def evaluate(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        self.model.eval()
        accuracies = []
        losses = []
        
        # Select appropriate training loader
        if self.dataset_name == 'reddit':
            train_loader = self.data_loader.get_client_dataloader(0)
        else:
            train_loader = self.train_loaders[0]
            
        loaders = [train_loader, self.val_loader, self.test_loader]
        
        for loader_idx, loader in enumerate(loaders):
            correct = total = 0
            running_loss = 0.0
            
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    data, target = batch[0], batch[1]
                else:
                    data, target = batch
                    
                data, target = data.to(self.device), target.to(self.device)
                
                if self.dataset_name == 'reddit':
                    # Reddit dataset specific handling
                    padding_mask = (data == loader.dataset.pad_idx)
                    output = self.model(data, src_padding_mask=padding_mask)
                    output = output.view(-1, output.size(-1))
                    target = target.view(-1)
                    
                    loss = self.criteria(output, target)
                    running_loss += loss.item() * target.size(0)
                    
                    pred = output.argmax(dim=1)
                    total += target.size(0)
                    
                    valid_mask = (target != loader.dataset.pad_idx) & (target != loader.dataset.unk_idx)
                    correct += ((pred == target) & valid_mask).sum().item()
                else:
                    output = self.model(data)
                    loss = self.criteria(output, target)
                    running_loss += loss.item() * data.size(0)
                    
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            if loader_idx == 0 and self.dataset_name == 'reddit':
                del train_loader
                torch.cuda.empty_cache()
            
            avg_loss = running_loss / total
            accuracy = 100. * correct / total
            
            accuracies.append(accuracy)
            losses.append(avg_loss)
        
        return (accuracies[0], losses[0]), (accuracies[1], losses[1]), (accuracies[2], losses[2])
# federated_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
import time

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
        self.eval_every = eval_every
        self.criteria = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        
    def train_round(self, round_num: int) -> Optional[Tuple[float, float, float]]:
        round_start_time = time.time() 
        torch.cuda.empty_cache()
        
        # # Add debug print at the start
        # print("\nModel parameter names:")
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: {param.shape}")

        # Get parameters that need perturbation
        params_to_perturb = []
        param_shapes = []
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'bias' in name:
                # print(f"{name}: {param.shape}")
                params_to_perturb.append(param)
                param_shapes.append(param.shape)
        
        # Compute center and range for each layer
        C, R = compute_center_and_range(self.model, self.device)
        global_state_dict = self.model.state_dict()
        
        # Initialize round-specific data
        round_data = self.method.initialize_round(len(self.train_loaders), self.model)
        client_ordering = round_data['client_ordering']  # Get the shuffled client order
        
        
        # Create mapping from parameter index to state_dict key
        param_to_key = {}
        idx = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'bias' in name:
                param_to_key[idx] = name
                idx += 1
        
        # Local training - process clients in batches
        local_models = []
        client_batch_size = 50
        n_clients = len(self.train_loaders)
        
        for client_batch_start in range(0, n_clients, client_batch_size):
            client_batch_end = min(client_batch_start + client_batch_size, n_clients)
            batch_updates = []
            
            for i in range(client_batch_start, client_batch_end):
                # client_model = type(self.model)().to(self.device)
                client_model = self.model_creator(self.device)
                client_model.load_state_dict(global_state_dict)
                # Use the correct dataset based on client ordering
                actual_client_idx = client_ordering[i]  # Get the actual client index

                # Train client model
                client_model = train_on_client(
                    client_model, 
                    self.train_loaders[actual_client_idx],
                    device=self.device,
                    lr = self.lr,
                    weight_decay = self.weight_decay,
                )
                
                # Convert model state dict for processing
                client_state = client_model.state_dict()
                
                
                # Process client update with correct parameter indexing
                processed_state_dict = self.method.process_client_update(
                    client_state,
                    i,
                    {param_to_key[idx]: c for idx, c in enumerate(C)},
                    {param_to_key[idx]: r for idx, r in enumerate(R)},
                    round_data
                )
                
                if processed_state_dict is not None:
                    batch_updates.append(processed_state_dict)
                
                del client_model
                torch.cuda.empty_cache()
            
            local_models.extend(batch_updates)
            del batch_updates
            torch.cuda.empty_cache()
        
        # Aggregate updates
        if local_models:
            global_state_dict = self.method.aggregate_updates(global_state_dict, local_models)
            self.model.load_state_dict(global_state_dict)
        
        del local_models
        torch.cuda.empty_cache()
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
        loaders = [self.train_loaders[0], self.val_loader, self.test_loader]
        
        for loader in loaders:
            correct = total = 0
            running_loss = 0.0
            
            for batch in loader:
                # Handle list format
                data, target = batch[0], batch[1]  # Extract from list
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.criteria(output, target)
                running_loss += loss.item() * data.size(0)  # Multiply by batch size
                
                # Compute accuracy
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
            # Calculate average loss and accuracy
            avg_loss = running_loss / total
            accuracy = 100. * correct / total
            
            accuracies.append(accuracy)
            losses.append(avg_loss)
        
        # Return (accuracy, loss) tuple for each loader: (train, val, test)
        return (accuracies[0], losses[0]), (accuracies[1], losses[1]), (accuracies[2], losses[2])
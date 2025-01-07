import torch
from .base import FederatedMethod
from main_dp_func import federated_learning_pairing

class FedAvg(FederatedMethod):
    def __init__(self, epsilon=None, num_rand=None, lambda_param=1, dropout=0, device='cuda'):
        super().__init__(epsilon=epsilon, num_rand=num_rand, lambda_param=lambda_param, 
                        dropout=dropout, device=device)
    
    def initialize_round(self, n_clients, global_model):
        """Initialize variables for the round"""
        self.n_clients = n_clients
        
        # Use the same federated_learning_pairing function for consistency
        client_ordering = federated_learning_pairing(n_clients)
        
        # All clients are active (1) by default
        client_roles = torch.ones(n_clients, device=self.device)
        
        # Apply dropout if specified
        if self.dropout > 0:
            dropout_mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[dropout_mask] = 2  # 2 represents dropped clients
        
        round_data = {
            'client_roles': client_roles,
            'client_ordering': client_ordering
        }
        
        return round_data
    
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process client update - for FedAvg, just return state dict if client is active"""
        client_role = round_data['client_roles'][client_idx]
        
        # Skip dropped out clients
        if client_role == 2:
            return None
        
        return client_state_dict
    
    def aggregate_updates(self, global_state_dict, local_models):
        """Aggregate updates using standard FedAvg averaging"""
        if not local_models:  # If no local models available
            return global_state_dict
            
        aggregated_state_dict = {}
        total_models = len(local_models)
        
        # Simple average of all parameters
        for key in global_state_dict.keys():
            if 'num_batches_tracked' in key:  # Skip batch norm statistics
                aggregated_state_dict[key] = global_state_dict[key]
                continue
            
            # Stack all local model parameters and take mean
            aggregated_state_dict[key] = torch.stack([
                local_state_dict[key] for local_state_dict in local_models
            ]).mean(dim=0)
        
        return aggregated_state_dict
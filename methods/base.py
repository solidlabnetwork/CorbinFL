from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class FederatedMethod(ABC):
    def __init__(self, epsilon=None, num_rand=None, lambda_param=1, dropout=0, device='cuda'):
        self.epsilon = epsilon
        self.num_rand = num_rand
        self.lambda_param = lambda_param
        self.dropout = dropout
        self.device = device
        self.n_clients = None  # Will be set during initialization
        
    @abstractmethod
    def initialize_round(self, n_clients, global_model):
        """Initialize method-specific variables for the round"""
        pass
        
    @abstractmethod
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process a single client's model update"""
        pass
    
    def aggregate_updates(self, global_state_dict, local_models):
        """Aggregate client updates - default weighted averaging"""
        for key in global_state_dict.keys():
            if global_state_dict[key].dtype == torch.long:
                mean_weights = torch.stack([local_model[key] for local_model in local_models], 0).float().mean(0).long()
            else:
                mean_weights = torch.stack([local_model[key] for local_model in local_models], 0).mean(0)
            
            global_state_dict[key] = (1 - self.lambda_param) * global_state_dict[key] + self.lambda_param * mean_weights
        return global_state_dict
# base.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import math

class FederatedMethod(ABC):
    def __init__(self, epsilon=None, num_rand=None, lambda_param=1, dropout=0, 
                 use_adam=False, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8, 
                 device='cuda'):
        self.epsilon = epsilon
        self.num_rand = num_rand
        self.lambda_param = lambda_param
        self.dropout = dropout
        self.device = device
        self.n_clients = None
        
        # Adam-related attributes
        self.use_adam = use_adam
        if use_adam:
            self.beta1 = beta1
            self.beta2 = beta2
            self.lr = lr
            self.eps = eps
            self.m = None  # First moment estimates
            self.v = None  # Second moment estimates
            self.t = 0     # Time step for bias correction

    def _initialize_adam(self, global_model):
        """Initialize Adam states if needed"""
        if self.use_adam and (self.m is None or self.v is None):
            self.m = {
                key: torch.zeros_like(param, device=self.device)
                for key, param in global_model.state_dict().items()
            }
            self.v = {
                key: torch.zeros_like(param, device=self.device)
                for key, param in global_model.state_dict().items()
            }
            self.t = 0  # Reset time step when initializing
        
    @abstractmethod
    def initialize_round(self, n_clients, global_model):
        """Initialize method-specific variables for the round"""
        self._initialize_adam(global_model)  # Always call this first
        pass
        
    @abstractmethod
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process a single client's model update"""
        pass
    
    def aggregate_updates(self, global_state_dict, local_models, weights=None):
        """
        Aggregate client updates using either lambda-based or Adam-style updates
        """
        if not local_models:
            return global_state_dict
            
        # Use equal weights if none provided
        if weights is None:
            weights = [1.0 / len(local_models)] * len(local_models)
        
        # Normalize weights
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.sum()
        
        # Calculate weighted average and update
        for key in global_state_dict.keys():
            if global_state_dict[key].dtype == torch.long:
                weighted_sum = torch.stack([
                    w * local_model[key].float() 
                    for w, local_model in zip(weights, local_models)
                ]).sum(0).long()
            else:
                weighted_sum = torch.stack([
                    w * local_model[key] 
                    for w, local_model in zip(weights, local_models)
                ]).sum(0)
            
            if self.use_adam:
                if key == list(global_state_dict.keys())[0]:
                    self.t += 1
                
                # Calculate relative update instead of raw difference
                relative_update = weighted_sum / (global_state_dict[key] + 1e-8) - 1.0
                
                # Update momentum estimates with scaled update
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * relative_update
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * relative_update.pow(2)
                
                # Bias correction
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # Apply scaled Adam update
                update_factor = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                global_state_dict[key] = global_state_dict[key] * (1 + update_factor)
            else:
                # Original lambda-based update
                global_state_dict[key] = (1 - self.lambda_param) * global_state_dict[key] + self.lambda_param * weighted_sum
        
        return global_state_dict
    
    def get_optimizer_state(self):
        """Return the current state of the optimizer"""
        if not self.use_adam:
            return None
            
        return {
            'iteration': self.t,
            'first_moment': self.m,
            'second_moment': self.v,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'lr': self.lr,
            'eps': self.eps
        }
    
    def load_optimizer_state(self, state):
        """Load a previously saved optimizer state"""
        if not self.use_adam or state is None:
            return
            
        self.t = state['iteration']
        self.m = state['first_moment']
        self.v = state['second_moment']
        # Optionally load hyperparameters if they were saved
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.lr = state.get('lr', self.lr)
        self.eps = state.get('eps', self.eps)
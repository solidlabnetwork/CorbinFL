import torch
from .base import FederatedMethod
import os
import numpy as np
from main_dp_func import federated_learning_pairing

class CorQuant(FederatedMethod):
    def __init__(self, epsilon, num_rand, lambda_param=1, dropout=0,
                 use_adam=False, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8,
                 device='cuda', save_weights=False, save_dir='saved_weights'):
        super().__init__(
            epsilon=epsilon,
            num_rand=num_rand,
            lambda_param=lambda_param,
            dropout=dropout,
            use_adam=use_adam,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            eps=eps,
            device=device
        )
        # self.alpha = (torch.exp(torch.tensor(epsilon)) + 1) / (torch.exp(torch.tensor(epsilon)) - 1)
        self.is_saved = False
        
    def initialize_round(self, n_clients, global_model):
        """Initialize variables for the round, including permutation tensors generation and client ordering"""
        self.n_clients = n_clients
        self._initialize_adam(global_model)
        
        # Use federated_learning_pairing to get client ordering
        client_ordering = federated_learning_pairing(n_clients)
        
        # Generate client assignments (2: dropout)
        client_roles = torch.zeros(n_clients, device=self.device)
        
        # Apply dropout if specified
        if self.dropout > 0:
            dropout_mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[dropout_mask] = 2
        
        # Generate permutation tensors for each parameter
        pi_dict = {}
        for name, param in global_model.named_parameters():
            if 'weight' in name or 'bias' in name:
                shape = param.shape
                # Create permutation tensor for each element in the parameter
                pi_dict[name] = torch.stack([
                    torch.randperm(n_clients, device=self.device) 
                    for _ in range(torch.prod(torch.tensor(shape)))
                ]).reshape(*shape, n_clients)
        
        round_data = {
            'client_roles': client_roles,
            'pi_dict': pi_dict,
            'client_ordering': client_ordering  # Add client ordering to round data
        }
        
        return round_data

    def _quantize_weight(self, W, pi, c, r, client_idx):
        """
        Implement quantization for a single weight tensor.
        
        Args:
            W (torch.Tensor): Weight tensor to quantize
            pi (torch.Tensor): Permutation tensor for the current client
            c (float): Center value
            r (float): Range value
            client_idx (int): Current client index
            
        Returns:
            torch.Tensor: Quantized weight tensor
        """
        # Get client-specific permutation
        pi_client = pi[..., client_idx]
        
        # Calculate range bounds
        l = c - r
        u = c + r
        
        # Clamp weights to range
        W = torch.clamp(W, min=l, max=u)
        
        # Calculate normalized position in range
        y = (W - l) / (2 * r)
        
        # Generate random offset
        gamma = torch.rand_like(W, device=self.device) / self.n_clients
        
        # Calculate U based on permutation and random offset
        U = (pi_client.float() / self.n_clients) + gamma
        
        # Determine quantized values
        Q = 2 * r * (U < y).float() + l
        
        return Q
    
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process client update according to CorQuant protocol"""
        client_role = round_data['client_roles'][client_idx]
        
        # Skip dropped out clients
        if client_role == 2:
            return None
            
        pi_dict = round_data['pi_dict']
        
        # Process each layer
        for key in client_state_dict.keys():
            if 'weight' in key or 'bias' in key:
                param_data = client_state_dict[key].data
                
                # Get permutation tensor for this layer
                pi = pi_dict[key]
                
                # Quantize weights using permutation-based scheme
                client_state_dict[key] = self._quantize_weight(
                    param_data,
                    pi,
                    C[key],  # Center value from dictionary
                    R[key],  # Range value from dictionary
                    client_idx
                )
                
        return client_state_dict
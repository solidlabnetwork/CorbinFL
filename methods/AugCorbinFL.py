import torch
from .base import FederatedMethod
import os
import numpy as np
from main_dp_func import federated_learning_pairing
from .corbin_fl import CorBinFL

class AugCorBinFL(CorBinFL):
    def __init__(self, epsilon, num_rand, gamma, lambda_param=1, dropout=0,
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
        self.gamma = gamma  # Fraction of clients using independent quantization
    
    def initialize_round(self, n_clients, global_model):
        """Initialize variables for the round, including client split and CR generation"""
        self.n_clients = n_clients
        self._initialize_adam(global_model)
        
        # Use federated_learning_pairing for client ordering
        client_ordering = federated_learning_pairing(n_clients)
        
        # Calculate number of clients for independent quantization
        n_independent = int(self.gamma * n_clients)
        
        # Split clients into independent and correlated groups
        independent_clients = set(client_ordering[:n_independent])
        correlated_clients = set(client_ordering[n_independent:])
        
        # Create pairs for correlated clients
        client_pairs = []
        remaining_correlated = list(correlated_clients)
        for i in range(0, len(remaining_correlated)-1, 2):
            client_pairs.append((remaining_correlated[i], remaining_correlated[i+1]))
        if len(remaining_correlated) % 2 != 0:
            client_pairs.append((remaining_correlated[-1], None))
        
        # Generate client roles (3: independent, 1: leader, 0: follower, 2: dropout)
        client_roles = torch.full((n_clients,), -1, device=self.device)
        
        # Assign independent roles
        for client in independent_clients:
            client_roles[client] = 3
            
        # Assign correlated roles
        for leader, follower in client_pairs:
            client_roles[leader] = 1
            if follower is not None:
                client_roles[follower] = 0
                    
        # Apply dropout if specified
        if self.dropout > 0:
            dropout_mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[dropout_mask] = 2
                
        # Generate CR for each leader in correlated group
        cr_dict = {}
        total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        for leader, _ in client_pairs:
            if client_roles[leader] == 1:  # Only generate CR if leader isn't dropped out
                cr_dict[leader] = self.comm_rand(self.num_rand, total_params, device=self.device)
        
        round_data = {
            'client_pairs': client_pairs,
            'client_roles': client_roles,
            'cr_dict': cr_dict,
            'client_ordering': client_ordering,
            'independent_clients': independent_clients
        }
        
        return round_data
    
    def _perturb_weight_independent(self, W, c, r):
        """
        Independent weight perturbation function (LDPFL-style).
        """
        shape = W.shape
        W = W.view(-1)
        
        # Calculate ProbMarginal
        ProbMarginal = 0.5 + (torch.clamp(W, c - r, c + r) - c) / (2 * self.alpha * r)
        
        # Apply independent noise
        U = torch.where(
            torch.rand_like(W) < ProbMarginal,
            torch.ones_like(W),
            -torch.ones_like(W)
        )
        
        # Update W
        W = c + U * self.alpha * r
        
        return W.view(shape)
    
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process client update according to client type (independent or correlated)"""
        client_role = round_data['client_roles'][client_idx]
        
        # Skip dropped out clients
        if client_role == 2:
            return None
            
        # Handle independent clients
        if client_role == 3:
            for key in client_state_dict.keys():
                if 'weight' in key or 'bias' in key:
                    param_data = client_state_dict[key].data
                    client_state_dict[key] = self._perturb_weight_independent(
                        param_data,
                        C[key],
                        R[key]
                    )
            return client_state_dict
            
        # Handle correlated clients (leader/follower)
        pair = next((pair for pair in round_data['client_pairs'] 
                    if client_idx in pair), None)
        if not pair:
            return None
            
        # Get CR and UP for correlated clients
        if client_role == 1:  # leader
            CR = round_data['cr_dict'].get(client_idx)
            UP = 1
        elif client_role == 0:  # follower
            UP = -1
            leader_idx = pair[0]
            if round_data['client_roles'][leader_idx] == 2:  # leader dropped out
                total_params = sum(v.numel() for k, v in client_state_dict.items() 
                                if 'weight' in k or 'bias' in k)
                CR = self.comm_rand(self.num_rand, total_params, self.device)
            else:
                CR = round_data['cr_dict'].get(leader_idx)
                if CR is None:
                    return None
        
        # Process each layer for correlated clients
        start_idx = 0
        for key in client_state_dict.keys():
            if 'weight' in key or 'bias' in key:
                param_data = client_state_dict[key].data
                num_elements = param_data.numel()
                
                CR_layer = CR[start_idx:start_idx + num_elements]
                
                client_state_dict[key] = self._perturb_weight(
                    param_data, 
                    C[key],
                    R[key],
                    CR_layer,
                    UP
                )
                    
                start_idx += num_elements
                
        return client_state_dict
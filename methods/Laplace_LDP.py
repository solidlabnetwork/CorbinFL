import torch
from .base import FederatedMethod
from main_dp_func import federated_learning_pairing


class LaplaceLDP(FederatedMethod):
    def __init__(self, epsilon, lambda_param=1, dropout=0,
                 use_adam=False, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8,
                 device='cuda'):
        super().__init__(
            epsilon=epsilon,
            lambda_param=lambda_param,
            dropout=dropout,
            use_adam=use_adam,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            eps=eps,
            device=device
        )
        self.alpha = (torch.exp(torch.tensor(epsilon)) + 1) / (torch.exp(torch.tensor(epsilon)) - 1)
        
    def initialize_round(self, n_clients, global_model):
        """Initialize variables for the round, including client pairing and CR generation"""
        self.n_clients = n_clients
        self._initialize_adam(global_model)
        # Use your existing federated_learning_pairing function
        client_ordering = federated_learning_pairing(n_clients)

        # Generate client assignments (1: active, 2: dropout)
        client_roles = torch.ones(n_clients, device=self.device)                    
        # Apply dropout if specified
        if self.dropout > 0:
            dropout_mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[dropout_mask] = 2
                

        # print('client roles:', client_roles)
        # print('client pairs:', client_pairs)           
        round_data = {
            'client_pairs': None,
            'client_roles': client_roles,
            'cr_dict': None,
            'client_ordering': client_ordering
        }
        
        return round_data

    def _perturb_weight(self, W, c, r):
        """
        Laplace mechanism weight perturbation with proper clipping
        """
        shape = W.shape
        W = W.view(-1)
        
        # First clip the weights
        W = torch.clamp(W, c - r, c + r)
        
        # Compute Laplace scale parameter (b = sensitivity/epsilon)
        sensitivity = 2 * r
        scale = sensitivity / self.epsilon
        
        # Add Laplace noise to clipped weights
        noise = torch.distributions.Laplace(0, scale).sample(W.shape).to(self.device)
        W = W + noise
        
        return W.view(shape)
    
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process client update according to LDP-FL protocol"""
        client_role = round_data['client_roles'][client_idx]
        
        # Skip dropped out clients
        if client_role == 2:
            return None
            
        # Process each layer
        for key in client_state_dict.keys():
            if 'weight' in key or 'bias' in key:
                param_data = client_state_dict[key].data
                
                # Use center and range values from the dictionaries
                client_state_dict[key] = self._perturb_weight(
                    param_data, 
                    C[key],  # Now accessing dictionary values
                    R[key]   # Now accessing dictionary values
                )
                
        return client_state_dict
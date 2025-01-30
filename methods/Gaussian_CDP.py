import torch
from .base import FederatedMethod
from main_dp_func import federated_learning_pairing


class GaussianCDP(FederatedMethod):
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
        self.alpha = (torch.exp(torch.tensor(epsilon)) + 1) / (torch.exp(torch.tensor(epsilon, device=self.device)) - 1)
        self.device = device
        
    def initialize_round(self, n_clients, global_model):
        """Initialize variables for the round, including client pairing and CR generation"""
        self._initialize_adam(global_model)
        # Use your existing federated_learning_pairing function
        client_ordering = federated_learning_pairing(n_clients)

        # Generate client assignments (1: active, 2: dropout)
        client_roles = torch.ones(n_clients, device=self.device)                    
        # Apply dropout if specified
        if self.dropout > 0:
            dropout_mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[dropout_mask] = 2
                
        round_data = {
            'client_pairs': None,
            'client_roles': client_roles,
            'cr_dict': None,
            'client_ordering': client_ordering
        }
        
        return round_data

    def _compute_sigma(self, r, client_roles):
        """
        Compute the Gaussian noise parameter sigma for CDP.
        sigma = 1/sqrt(n) * r * alpha
        where alpha = (exp(epsilon)+1)/(exp(epsilon)-1)
        and n is the number of active clients
        """
        # Calculate number of active clients (role <= 1)
        n = torch.sum(client_roles <= 1).float()
        
        # Calculate sigma based on the CDP formula
        sigma = (1.0 / torch.sqrt(n)) * r * self.alpha
        return sigma

    def _perturb_weight(self, W, c, r, client_roles):
        """
        Gaussian mechanism weight perturbation with CDP noise calibration.
        """
        shape = W.shape
        W = W.view(-1)
        
        # First clip the weights to establish exact sensitivity
        W = torch.clamp(W, c - r, c + r)
        
        # Compute sigma using the CDP formula
        sigma = self._compute_sigma(r, client_roles)
        
        # Add Gaussian noise to clipped weights
        noise = torch.normal(0, sigma, size=W.shape, device=self.device)
        W = W + noise
        
        return W.view(shape)
    
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process client update according to Gaussian CDP protocol"""
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
                    R[key],  # Now accessing dictionary values
                    round_data['client_roles']
                )
                
        return client_state_dict
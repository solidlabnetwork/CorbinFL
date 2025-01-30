import torch
from .base import FederatedMethod
from main_dp_func import federated_learning_pairing
import math
from scipy import optimize


class GaussianLDP(FederatedMethod):
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

    def compute_sigma_agm(self,epsilon, delta, sensitivity):
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
    
    def _perturb_weight(self, W, c, r):
        """
        Gaussian mechanism weight perturbation with proper clipping.
        """
        shape = W.shape
        W = W.view(-1)
        
        # First clip the weights to establish exact sensitivity
        W = torch.clamp(W, c - r, c + r)
        
        # Compute sensitivity and noise parameters
        delta = 1e-5
        sensitivity = 2 * r
        # sigma = sensitivity * torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25/delta, device=self.device)), device=self.device)) / self.epsilon
        # sigma = sensitivity * torch.sqrt(2 * torch.log(torch.tensor(1.25/delta, device=self.device))) / self.epsilon
        sigma = self.compute_sigma_agm(self.epsilon, delta, sensitivity)
        # print('sigma:', sigma)
        # print('sensitivity:', sensitivity)
        # print('epsilon:', self.epsilon)
        # print('delta:', delta)
        # Add Gaussian noise to clipped weights
        noise = torch.normal(0, sigma, size=W.shape, device=self.device)
        W = W + noise
        
        return W.view(shape)
    
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process client update according to Gaussian protocol"""
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
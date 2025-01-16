# Distributed SignNum
# Bernstein, Jeremy, et al. "signSGD with Majority Vote is Communication Efficient and Fault Tolerant." 
# International Conference on Learning Representations. 2019
import torch
from .base import FederatedMethod

class SignSGD(FederatedMethod):
    def __init__(self, lr=0.0001, beta=0.9, weight_decay=0, dropout=0, device='cuda'):
        super().__init__(dropout=dropout, device=device)
        self.lr = lr  # η in the algorithm
        self.beta = beta  # β in the algorithm
        self.weight_decay = weight_decay  # λ in the algorithm
        
    def initialize_round(self, n_clients, global_model):
        self.n_clients = n_clients
        
        # Generate client roles (1 for active, 2 for dropped)
        client_roles = torch.ones(n_clients, device=self.device)
        if self.dropout > 0:
            dropout_mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[dropout_mask] = 2
            
        return {
            'client_roles': client_roles,
            'client_ordering': list(range(n_clients))
        }
    
    def process_client_update(self, client_momentum_signs, client_idx, C, R, round_data):
        """Process client update (which is now signs of momentum from client)"""
        if round_data['client_roles'][client_idx] == 2:  # Skip dropped clients
            return None
            
        # Client already computed momentum and its sign, just return it
        return client_momentum_signs
    
    def aggregate_updates(self, global_state_dict, local_momentum_signs, weights=None):
        """
        Aggregate according to Algorithm 1:
        1. Sum the momentum signs from all clients
        2. Take sign of the sum
        3. Update including weight decay term
        """
        if not local_momentum_signs:
            return global_state_dict
            
        for key in global_state_dict:
            if 'weight' in key or 'bias' in key:
                # V ← ∑sign(vm)
                V = local_momentum_signs[0][key]  # Sum is already computed in federated trainer
                
                # sign(V)
                aggregated_sign = torch.sign(V)
                
                # x ← x - η(sign(V) + λx)
                global_state_dict[key] = global_state_dict[key] - self.lr * (
                    aggregated_sign + self.weight_decay * global_state_dict[key]
                )
        
        return global_state_dict
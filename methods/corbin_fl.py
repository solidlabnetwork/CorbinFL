# corbin_fl.py
import torch
from .base import FederatedMethod
import os
import numpy as np
from main_dp_func import federated_learning_pairing


class CorBinFL(FederatedMethod):
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
        self.alpha = (torch.exp(torch.tensor(epsilon)) + 1) / (torch.exp(torch.tensor(epsilon)) - 1)
        self.is_saved = False
        
    def initialize_round(self, n_clients, global_model):
        """Initialize variables for the round, including client pairing and CR generation"""
        self.n_clients = n_clients
        self._initialize_adam(global_model)
        # Use your existing federated_learning_pairing function
        client_ordering = federated_learning_pairing(n_clients)
        # print(f"New client ordering this round: {client_ordering}")
        # Create pairs based on the random ordering
        client_pairs = []
        for i in range(0, n_clients-1, 2):
            client_pairs.append((client_ordering[i], client_ordering[i+1]))
        if n_clients % 2 != 0:
            client_pairs.append((client_ordering[-1], None))
        
        # print(f"Pairs this round: {client_pairs}")
        # Generate client assignments (1: leader, 0: follower, 2: dropout)
        client_roles = torch.ones(n_clients, device=self.device)
        for leader, follower in client_pairs:
            if follower is not None:
                client_roles[follower] = 0
                    
        # Apply dropout if specified
        if self.dropout > 0:
            dropout_mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[dropout_mask] = 2
                
        # Generate CR for each leader
        cr_dict = {}
        total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        for leader, _ in client_pairs:
            if client_roles[leader] == 1:  # Only generate CR if leader isn't dropped out
                cr_dict[leader] = self.comm_rand(self.num_rand, total_params, device=self.device)
        # print('client roles:', client_roles)
        # print('client pairs:', client_pairs)           
        round_data = {
            'client_pairs': client_pairs,
            'client_roles': client_roles,
            'cr_dict': cr_dict,
            'client_ordering': client_ordering
        }
        
        return round_data

    def comm_rand(self,NumRand, NumParam, device='cpu'):
        return torch.randint(0, 2 ** NumRand, (NumParam,), device=device)
    
    def _find_index_batch(self, NumRand, P_batch):
        """
        Find indices for a batch of probabilities.
        """
        two_pow_d = 2 ** NumRand
        indices = torch.floor(P_batch * two_pow_d).long()
        indices = torch.clamp(indices, 0, two_pow_d - 1)
        return indices.to(P_batch.device)

    def save_layer_probabilities(self, probabilities, client_idx, layer_name, round_num):
        """
        Save probabilities for a specific layer to a file.
        
        Args:
            probabilities (torch.Tensor): Probability tensor to save
            client_idx (int): Client identifier
            layer_name (str): Name of the layer
            round_num (int): Current communication round
        """
        # Convert to numpy for easier saving
        probs_np = probabilities.cpu().numpy()
        
        # Create filename with round and client info
        filename = f"round_{round_num}_client_{client_idx}_{layer_name}_probs.npy"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save the numpy array
        np.save(filepath, probs_np)
    def _perturb_weight(self, W, c, r, CR, UP):
        """
        CorBinFL-specific weight perturbation function.
        
        Args:
            W (torch.Tensor): Weight tensor to perturb
            c (float): Center value
            r (float): Range value
            CR (torch.Tensor): Common randomness tensor
            UP (int): Direction indicator (1 for leader, -1 for follower)
            
        Returns:
            torch.Tensor: Perturbed weight tensor
        """
        shape = W.shape
        W = W.view(-1)
        
        # Calculate ProbMarginal
        ProbMarginal = 0.5 + (torch.clamp(W, c - r, c + r) - c) / (2 * self.alpha * r)
        # self.save_layer_probabilities(
        #                 ProbMarginal,
        #                 client_idx,
        #                 key,
        #                 self.current_round
        #             )

        # Calculate P based on UP
        P = ProbMarginal if UP == 1 else 1 - ProbMarginal
        
        # Find indices using P
        idx = self._find_index_batch(self.num_rand, P)
        IntCR = CR.long()
        
        # Create initial U tensor
        U = torch.full_like(W, UP, dtype=torch.float32)
        U = torch.where(IntCR >= idx + 1, -UP, U)
        
        # Handle middle cases
        middle_case = (IntCR >= idx) & (IntCR < idx + 1)
        
        # Handle edge cases
        idx = torch.where(idx + 1 >= 2**self.num_rand, 2**self.num_rand - 2, idx)
        
        if middle_case.any():
            # Calculate probabilities for idx
            prob_idx = idx.float() / 2**self.num_rand
            
            # Calculate ProbGrid for middle case
            ProbGrid = torch.zeros_like(W)
            ProbGrid[middle_case] = (P[middle_case] - prob_idx[middle_case]) * 2**self.num_rand
            
            # Generate random values and apply to middle case
            random_values = torch.rand_like(W)
            UP_tensor = torch.tensor(UP, dtype=torch.float32, device=self.device)
            U[middle_case] = torch.where(random_values[middle_case] < ProbGrid[middle_case],
                                       UP_tensor,
                                       -UP_tensor)
        
        # Update W for all elements at once
        W = c + U * self.alpha * r
        
        return W.view(shape)
    
    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        """Process client update according to CorBinFL protocol"""
        client_role = round_data['client_roles'][client_idx]
        
        # Skip dropped out clients
        if client_role == 2:
            return None
            
        # Find this client's pair
        pair = next((pair for pair in round_data['client_pairs'] 
                    if client_idx in pair), None)
        if not pair:
            return None
            
        # Handle CR generation
        if client_role == 1:  # if the client is leader
            total_params = sum(v.numel() for k, v in client_state_dict.items() 
                            if 'weight' in k or 'bias' in k)
            # CR = self.comm_rand(self.num_rand, total_params, self.device)
            # round_data['cr_dict'][client_idx] = CR
            CR = round_data['cr_dict'].get(client_idx)
            UP = 1
        elif client_role == 0:  # if the client is follower
            UP = -1
            leader_idx = pair[0]
            if round_data['client_roles'][leader_idx] == 2:  # if the leader dropped out
                # Create new CR for follower with dropped out leader
                total_params = sum(v.numel() for k, v in client_state_dict.items() 
                                if 'weight' in k or 'bias' in k)
                CR = self.comm_rand(self.num_rand, total_params, self.device) # only generate if the leader dropped out

            else:
                CR = round_data['cr_dict'].get(leader_idx)
                if CR is None:
                    return None
            
        # Process each layer
        start_idx = 0
        for key in client_state_dict.keys():
            if 'weight' in key or 'bias' in key:
                param_data = client_state_dict[key].data
                num_elements = param_data.numel()
                
                CR_layer = CR[start_idx:start_idx + num_elements]
                
                # Use center and range values from the dictionaries
                client_state_dict[key] = self._perturb_weight(
                    param_data, 
                    C[key],  # Now accessing dictionary values
                    R[key],  # Now accessing dictionary values
                    CR_layer,
                    UP
                )
                    
                start_idx += num_elements
                
        return client_state_dict
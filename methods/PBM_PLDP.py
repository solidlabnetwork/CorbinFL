import torch
import numpy as np
from .base import FederatedMethod
from main_dp_func import federated_learning_pairing


class PBM_PLDP(FederatedMethod):
    """
    Privatized Binomial Mechanism with Pure LDP (PBM-PLDP).
    Each coordinate is encoded with m iid Bernoulli trials; no shared randomness.
      m=1 -> 1 bit per coordinate, theta = 0.5 * tanh(eps/2)
      m=2 -> 2 bits per coordinate, theta = 0.5 * tanh(eps/4)
    Satisfies the same eps-PLDP condition (C2/C5) as CorBin-FL by construction.
    """

    def __init__(self, epsilon, m=1, lambda_param=1, dropout=0,
                 use_adam=False, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8,
                 device='cuda', r_max=None):
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
        self.m = m
        self.r_max = r_max  # cap on encoding range; None = no cap
        # theta calibrated so max likelihood ratio across all outputs equals e^eps
        self.theta = 0.5 * float(np.tanh(epsilon / (2.0 * m)))

    def initialize_round(self, n_clients, global_model):
        self.n_clients = n_clients
        self._initialize_adam(global_model)
        client_ordering = federated_learning_pairing(n_clients)
        client_roles = torch.ones(n_clients, device=self.device)
        if self.dropout > 0:
            mask = torch.rand(n_clients, device=self.device) < self.dropout
            client_roles[mask] = 2
        return {
            'client_pairs': None,
            'client_roles': client_roles,
            'cr_dict': None,
            'client_ordering': client_ordering,
        }

    def _perturb_weight(self, W, c, r, debug=False, layer_name=''):
        c_val = c.item() if isinstance(c, torch.Tensor) else float(c)
        r_val = r.item() if isinstance(r, torch.Tensor) else float(r)
        if r_val < 1e-8 or self.theta < 1e-12:
            return W
        if self.r_max is not None:
            r_val = min(r_val, self.r_max)

        shape = W.shape
        W_flat = W.view(-1).float()

        # p in [0.5-theta, 0.5+theta]
        W_clipped = torch.clamp(W_flat, c_val - r_val, c_val + r_val)
        p = 0.5 + (self.theta / r_val) * (W_clipped - c_val)
        p = torch.clamp(p, 0.0, 1.0)

        # Z ~ Binomial(m, p); each client uses only its own coins
        Z = torch.distributions.Binomial(total_count=self.m, probs=p).sample()

        # unbiased decode: E[w_hat] = w
        W_hat = c_val + (r_val / self.theta) * (Z / self.m - 0.5)

        if debug:
            amp = r_val / self.theta
            print(f"  [{layer_name}] r={r_val:.4f} theta={self.theta:.4f} amp={amp:.2f} "
                  f"W_hat range=[{W_hat.min().item():.3f}, {W_hat.max().item():.3f}]")

        return W_hat.view(shape)

    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        if round_data['client_roles'][client_idx] == 2:
            return None

        debug = (client_idx == 0)
        if debug:
            print(f"[PBM_PLDP debug] client 0:")
        for key in client_state_dict.keys():
            if 'weight' in key or 'bias' in key:
                client_state_dict[key] = self._perturb_weight(
                    client_state_dict[key].data,
                    C[key],
                    R[key],
                    debug=debug,
                    layer_name=key,
                )

        return client_state_dict

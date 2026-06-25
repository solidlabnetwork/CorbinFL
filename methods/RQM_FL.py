import torch
import numpy as np
from scipy.optimize import brentq
from .base import FederatedMethod
from main_dp_func import federated_learning_pairing
from RQM import rqm_local_pure_dp_eps


class RQM_FL(FederatedMethod):
    """
    Randomized Quantization Mechanism (RQM) for federated learning.
    Youn, Hu, Ziani, Abernethy (2023), Algorithm 2.

    m levels -> ceil(log2(m)) bits per coordinate. Supported: m=2 (1 bit), m=3 (2 bits).
    No shared randomness — each client uses only its own private coins.

    Calibration: given (epsilon, m, q), Delta is solved at construction via bisection
    on rqm_local_pure_dp_eps(1.0, delta_ratio, m, q) = epsilon, exploiting the
    scale-invariance of epsilon in Delta/c. At encode time: Delta = delta_ratio * R[key].

    Note: q has no effect when m=2 (no interior levels exist); it is accepted for
    API consistency and to allow the same CLI interface for both m values.
    """

    def __init__(self, epsilon, m=2, q=0.5, lambda_param=1, dropout=0,
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
        if m not in (2, 3):
            raise ValueError("RQM_FL only supports m=2 (1-bit) or m=3 (2-bit).")
        self.m = m
        self.q = float(q)
        self.r_max = r_max  # cap on encoding range; None = no cap

        # Calibrate delta_ratio: eps(1.0, delta_ratio, m, q) = epsilon.
        # eps is monotone decreasing in delta_ratio, so brentq converges reliably.
        # For m=2: closed form is delta_ratio = 2/(exp(eps)-1), bisection gives same.
        f = lambda dr: rqm_local_pure_dp_eps(1.0, dr, m, q) - epsilon
        self.delta_ratio = brentq(f, 1e-6, 1e3, xtol=1e-9)
        print(f"[RQM] m={m}, q={q}, epsilon={epsilon} -> delta_ratio={self.delta_ratio:.6f}")

    # ------------------------------------------------------------------
    # FederatedMethod interface
    # ------------------------------------------------------------------

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

    def _perturb_weight(self, W, c, r):
        c_val = c.item() if isinstance(c, torch.Tensor) else float(c)
        r_val = r.item() if isinstance(r, torch.Tensor) else float(r)
        if r_val < 1e-8:
            return W
        if self.r_max is not None:
            r_val = min(r_val, self.r_max)

        shape = W.shape
        W_flat = W.view(-1).float()

        # Center and clip to [-r, r]
        x = torch.clamp(W_flat, c_val - r_val, c_val + r_val) - c_val

        Xmax = r_val * (1.0 + self.delta_ratio)

        if self.m == 2:
            # Levels: B = [-Xmax, +Xmax]
            # p_up = (x - B[0]) / (B[1] - B[0]) = (x + Xmax) / (2 * Xmax)
            p_up = torch.clamp((x + Xmax) / (2.0 * Xmax), 0.0, 1.0)
            z = torch.bernoulli(p_up)                         # in {0, 1}
            # decode: B[z] = -Xmax + z * 2 * Xmax
            W_hat = c_val + (-Xmax + z * 2.0 * Xmax)

        else:  # m == 3
            # Levels: B = [-Xmax, 0, +Xmax] (B[1] = 0 always)
            # Interior level (index 1) kept per-coordinate with prob q.
            keep = torch.bernoulli(torch.full_like(x, self.q))  # 1 = kept, 0 = dropped

            # --- not-kept branch: bracket always (0, 2) ---
            p_up_nk = torch.clamp((x + Xmax) / (2.0 * Xmax), 0.0, 1.0)
            z_nk = torch.bernoulli(p_up_nk) * 2.0             # in {0.0, 2.0}

            # --- kept branch: bracket depends on sign of x ---
            # x < 0: bracket (0,1), p_up = (x + Xmax) / Xmax
            # x >= 0: bracket (1,2), p_up = x / Xmax
            is_low = x < 0
            p_up_k = torch.where(is_low,
                                 (x + Xmax) / Xmax,
                                 x / Xmax)
            p_up_k = torch.clamp(p_up_k, 0.0, 1.0)
            round_k = torch.bernoulli(p_up_k)
            z_k = torch.where(is_low, round_k, 1.0 + round_k)  # {0,1} or {1,2}

            z = torch.where(keep > 0.5, z_k, z_nk)
            # decode: B[z] = -Xmax + z * Xmax  (spacing = Xmax for m=3)
            W_hat = c_val + (-Xmax + z * Xmax)

        return W_hat.view(shape)

    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        if round_data['client_roles'][client_idx] == 2:
            return None

        for key in client_state_dict.keys():
            if 'weight' in key or 'bias' in key:
                client_state_dict[key] = self._perturb_weight(
                    client_state_dict[key].data,
                    C[key],
                    R[key],
                )

        return client_state_dict

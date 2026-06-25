# Interpolated Minimum Variance Unbiased (I-MVU) mechanism for federated learning.
#
# Reference: Guo et al., "Privacy-Aware Compression for Federated Learning Through
# Numerical Mechanism Design", ICML 2023.
# GitHub: https://github.com/facebookresearch/dp_compression
#
# Runtime mechanism (follows IMVUMechanismPyTorch from the reference codebase):
#   Given a precomputed transition matrix P (B_in × B_out) and alpha (B_out,),
#   for a continuous input x in [0,1] we linearly interpolate between adjacent rows
#   of log(P) to obtain output log-probabilities, then sample a symbol and decode.
#
#   k = floor((B_in - 1) * x)                  # lower quantization index
#   eta = coeff1 * log(P[k,:]) + coeff2 * log(P[k+1,:])   # interpolated log-probs
#   z   ~ categorical(softmax(eta))             # sampled symbol
#   out = alpha[z]                              # reconstructed value in [~0, ~1]

import torch
import torch.nn.functional as F
import numpy as np
import math
import os

from .base import FederatedMethod
from main_dp_func import federated_learning_pairing


class IMVU(FederatedMethod):

    def __init__(self, epsilon, budget=1, beta=1.0, lambda_param=1, dropout=0,
                 use_adam=False, beta1=0.9, beta2=0.999, lr=0.001, eps_adam=1e-8,
                 imvu_params_dir='imvu_params', device='cuda'):
        super().__init__(
            epsilon=epsilon,
            lambda_param=lambda_param,
            dropout=dropout,
            use_adam=use_adam,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            eps=eps_adam,
            device=device
        )
        self.budget = budget
        self.beta = beta

        P_mat, alpha_vals, input_bits = self._load_or_compute_params(
            epsilon, budget, imvu_params_dir
        )

        self.input_bits = input_bits
        self.B_in = 2 ** input_bits

        # eta: log(P), shape (B_in, B_out) — stored on device for fast inference
        # clamp before log so zero entries (from small-epsilon numerical underflow)
        # become -69 instead of -inf, keeping softmax well-defined downstream
        self.eta = torch.log(P_mat.clamp(min=1e-30)).to(device)   # (B_in, B_out)
        self.alpha_tensor = alpha_vals.to(device)        # (B_out,)

    # ------------------------------------------------------------------
    # Parameter loading / computation
    # ------------------------------------------------------------------

    def _load_or_compute_params(self, epsilon, budget, params_dir):
        fname = os.path.join(params_dir, f'imvu_eps{epsilon}_b{budget}.pt')

        if os.path.exists(fname):
            data = torch.load(fname, map_location='cpu', weights_only=False)
            print(f"[IMVU] Loaded params from {fname}")
            # Support both old (1D p) and new (2D P) file formats
            if 'P' in data:
                P = data['P']                              # (B_in, B_out)
                alpha = data['alpha']                      # (B_out,)
                input_bits = int(data.get('input_bits', int(math.log2(P.shape[0]))))
            else:
                # Legacy 1D format: treat p as a (2, B_out) P matrix
                p = data['p']                              # (B_out,)
                alpha = data['alpha']
                P = torch.stack([p, p.flip(0)], dim=0)    # (2, B_out)
                input_bits = 1
            return P, alpha, input_bits

        # No precomputed file — compute inline and warn
        print(f"[IMVU] Precomputed file not found: {fname}")
        print(f"[IMVU] Run first:  python precompute_imvu.py "
              f"--epsilon {epsilon} --budget {budget}")
        print(f"[IMVU] Computing on-the-fly (may take a few seconds)...")
        P_np, alpha_np, input_bits = self._compute_params(epsilon, budget)
        return (torch.tensor(P_np, dtype=torch.float32),
                torch.tensor(alpha_np, dtype=torch.float32),
                input_bits)

    def _compute_params(self, epsilon, budget, input_bits=3):
        """Full MVU trust-region optimization (scipy, no cvxpy). Used as fallback."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from precompute_imvu import compute_mvu_params

        P, alpha = compute_mvu_params(epsilon, budget, input_bits)
        return P, alpha, input_bits

    # ------------------------------------------------------------------
    # Core interpolation and sampling (follows IMVUMechanismPyTorch.get_etas)
    # ------------------------------------------------------------------

    def _get_etas(self, x):
        """
        Linearly interpolate between adjacent rows of log(P) for inputs x in [0,1].

        x:   FloatTensor of shape (N,), values in [0, 1]
        Returns: FloatTensor of shape (N, B_out), interpolated log-probabilities.
        """
        B = self.B_in
        # Quantization: find floor index k and fractional part
        k = torch.floor((B - 1) * x).long().clamp(0, B - 2)
        coeff2 = (B - 1) * x - k.float()   # ∈ [0, 1)
        coeff1 = 1.0 - coeff2

        eta_lo = self.eta[k]                         # (N, B_out) — row k
        eta_hi = self.eta[(k + 1).clamp(max=B - 1)]  # (N, B_out) — row k+1
        return coeff1[:, None] * eta_lo + coeff2[:, None] * eta_hi

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
            'client_ordering': client_ordering
        }

    def _perturb_weight(self, W, c, r):
        """
        Apply I-MVU perturbation in weight space.

        1. Normalize W to [0, 1] using layer bounds [c-r, c+r].
        2. Interpolate log-probabilities between rows of P.
        3. Sample a discrete symbol from the resulting distribution.
        4. Decode with alpha, then denormalize back to original range.
        """
        r_val = r.item() if isinstance(r, torch.Tensor) else float(r)
        if r_val < 1e-8:
            return W

        shape = W.shape
        W_flat = W.view(-1).float()

        # β-scaled normalization (Section 3.1): x = 0.5 + β*(w - c) / (2r)
        # Maps [c-r, c+r] to [(1-β)/2, (1+β)/2]; β>1 spreads concentrated inputs.
        # No final clamp — extrapolation beyond [0,1] is valid and intentional.
        W_clipped = torch.clamp(W_flat, c - r, c + r)
        x = 0.5 + self.beta * (W_clipped - c) / (2.0 * r)

        # Interpolated output distribution
        log_probs = self._get_etas(x)                   # (N, B_out)
        # nan_to_num guards the rare case where a degenerate P row (all -inf after
        # interpolation) causes softmax to produce NaN; fallback is uniform over symbols
        probs = torch.softmax(log_probs, dim=1).nan_to_num(nan=1.0 / log_probs.shape[1]).clamp(0.0, 1.0)

        # Sample symbol
        if self.budget == 1:
            symbols = torch.bernoulli(probs[:, 1]).long()
        else:
            symbols = torch.multinomial(probs, 1, replacement=True).squeeze(-1)

        # Decode: alpha lives in [0,1] space; invert the β-scaling to get back to weight space.
        # x = 0.5 + β*(w-c)/(2r)  →  w = (x - 0.5) * (2r/β) + c
        x_recon = self.alpha_tensor[symbols]
        W_new = (x_recon - 0.5) * (2.0 * r / self.beta) + c
        return W_new.view(shape)

    def process_client_update(self, client_state_dict, client_idx, C, R, round_data):
        if round_data['client_roles'][client_idx] == 2:
            return None

        for key in client_state_dict.keys():
            if 'weight' in key or 'bias' in key:
                client_state_dict[key] = self._perturb_weight(
                    client_state_dict[key].data,
                    C[key],
                    R[key]
                )

        return client_state_dict

"""
Precompute I-MVU / MVU mechanism parameters for federated learning experiments.

Uses the full MVU (Minimum Variance Unbiased) trust-region optimization from:
  Guo et al., "Privacy-Aware Compression for Federated Learning Through Numerical
  Mechanism Design", ICML 2023. GitHub: facebookresearch/dp_compression

The optimization finds a transition matrix P (B_in × B_out) and reconstruction
alphabet alpha (B_out,) that are:
  - epsilon-LDP: P[i,j] / P[i',j] <= e^epsilon  for all i,i',j
  - Unbiased:    sum_j P[i,j]*alpha[j] = i/(B_in-1)  for all i
  - Min MSE:     minimize (1/B_in) * sum_{i,j} P[i,j] * (i/(B_in-1) - alpha[j])^2

At runtime the IMVU method linearly interpolates between adjacent rows of log(P)
to handle continuous inputs x in [0,1] (the InterpolatedMVU extension).

Only scipy is required — no cvxpy.

Usage:
    python precompute_imvu.py --epsilon 1 3 5 7 --budget 1 2 4
    python precompute_imvu.py --epsilon 5 --budget 4 --input_bits 4 --verbose
"""

import numpy as np
from scipy import optimize, sparse
import torch
import argparse
import os
import math


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def _build_dp_constraint(B_in, B_out, epsilon, n_vars):
    """
    Strict epsilon-LDP constraint: P[i,j] - e^eps * P[i',j] <= 0  for all i != i', j.
    Returns a scipy LinearConstraint acting on the full variable vector [P.flat, alpha].
    """
    exp_eps = math.exp(epsilon)
    rows, cols, data = [], [], []
    row_idx = 0
    for i in range(B_in):
        for i2 in range(B_in):
            if i == i2:
                continue
            for j in range(B_out):
                rows += [row_idx, row_idx]
                cols += [i * B_out + j, i2 * B_out + j]
                data += [1.0, -exp_eps]
                row_idx += 1
    A = sparse.csr_matrix((data, (rows, cols)), shape=(row_idx, n_vars))
    return optimize.LinearConstraint(A, -np.inf, np.zeros(row_idx))


def _build_row_stochastic_constraint(B_in, B_out, n_vars):
    """
    Row-stochastic constraint: sum_j P[i,j] = 1  for all i.
    Returns a scipy LinearConstraint.
    """
    rows, cols, data = [], [], []
    for i in range(B_in):
        for j in range(B_out):
            rows.append(i)
            cols.append(i * B_out + j)
            data.append(1.0)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(B_in, n_vars))
    return optimize.LinearConstraint(A, 1.0, 1.0)


def _build_unbiased_constraint(B_in, B_out, n_vars):
    """
    Unbiasedness constraint: P @ alpha = target  for all rows i.
    This is nonlinear in (P, alpha) and handled with NonlinearConstraint.
    """
    n_P = B_in * B_out
    target = np.arange(B_in, dtype=float) / (B_in - 1)

    def fn(x):
        P = x[:n_P].reshape(B_in, B_out)
        alpha = x[n_P:]
        return P @ alpha - target                          # shape (B_in,)

    def jac(x):
        P = x[:n_P].reshape(B_in, B_out)
        alpha = x[n_P:]
        rows_j, cols_j, data_j = [], [], []
        for i in range(B_in):
            for j in range(B_out):
                # d c_i / d P[i,j] = alpha[j]
                rows_j.append(i);  cols_j.append(i * B_out + j);  data_j.append(alpha[j])
                # d c_i / d alpha[j] = P[i,j]
                rows_j.append(i);  cols_j.append(n_P + j);         data_j.append(P[i, j])
        return sparse.csr_matrix((data_j, (rows_j, cols_j)), shape=(B_in, n_vars))

    def hess(x, v):
        # Mixed P-alpha second derivatives only
        rows_h, cols_h, data_h = [], [], []
        for i in range(B_in):
            for j in range(B_out):
                # d^2 c_i / (d P[i,j] d alpha[j]) = 1
                rows_h += [i * B_out + j, n_P + j]
                cols_h += [n_P + j,       i * B_out + j]
                data_h += [v[i], v[i]]
        return sparse.csr_matrix((data_h, (rows_h, cols_h)), shape=(n_vars, n_vars))

    return optimize.NonlinearConstraint(fn, 0.0, 0.0, jac=jac, hess=hess)


def compute_mvu_params(epsilon, budget, input_bits=3, verbose=False):
    """
    Compute MVU mechanism parameters P (B_in × B_out) and alpha (B_out,).

    Args:
        epsilon:    Privacy budget.
        budget:     Output bits per coordinate (B_out = 2^budget).
        input_bits: Input quantization level (B_in = 2^input_bits).
                    Larger → better MSE but slower optimization. 3 (=8 levels) is
                    a good default matching the dp_compression paper's results.
        verbose:    Print optimization progress.

    Returns:
        P:     numpy array (B_in, B_out), transition probability matrix.
        alpha: numpy array (B_out,), reconstruction values (roughly in [0,1]).
    """
    B_in = 2 ** input_bits
    B_out = 2 ** budget
    n_P = B_in * B_out
    n_alpha = B_out
    n = n_P + n_alpha
    target = np.arange(B_in, dtype=float) / (B_in - 1)

    # --- Objective and derivatives ---
    def objective(x):
        P = x[:n_P].reshape(B_in, B_out)
        alpha = x[n_P:]
        return (P * (target[:, None] - alpha[None, :]) ** 2).sum() / B_in

    def obj_jac(x):
        P = x[:n_P].reshape(B_in, B_out)
        alpha = x[n_P:]
        g = np.zeros(n)
        g[:n_P] = ((target[:, None] - alpha[None, :]) ** 2).reshape(n_P) / B_in
        for j in range(B_out):
            g[n_P + j] = -2.0 * np.dot(P[:, j], target - alpha[j]) / B_in
        return g

    def obj_hess(x):
        P = x[:n_P].reshape(B_in, B_out)
        alpha = x[n_P:]
        rows_h, cols_h, data_h = [], [], []
        for i in range(B_in):
            for j in range(B_out):
                v = -2.0 * (target[i] - alpha[j]) / B_in
                rows_h += [i * B_out + j, n_P + j]
                cols_h += [n_P + j,       i * B_out + j]
                data_h += [v, v]
        for j in range(B_out):
            rows_h.append(n_P + j); cols_h.append(n_P + j)
            data_h.append(2.0 * P[:, j].sum() / B_in)
        return sparse.csr_matrix((data_h, (rows_h, cols_h)), shape=(n, n))

    # --- Constraints ---
    c_dp   = _build_dp_constraint(B_in, B_out, epsilon, n)
    c_row  = _build_row_stochastic_constraint(B_in, B_out, n)
    c_unb  = _build_unbiased_constraint(B_in, B_out, n)

    # --- Bounds: P >= 0, alpha unconstrained ---
    lb = np.concatenate([np.zeros(n_P), -np.inf * np.ones(n_alpha)])
    ub = np.concatenate([np.ones(n_P),   np.inf * np.ones(n_alpha)])
    bounds = optimize.Bounds(lb, ub)

    # --- Initial point ---
    # Uniform P, alpha linearly spaced matching the known range from budget=1
    s = math.exp(epsilon)
    a0_min = -1.0 / (s - 1) if s > 1 else -1.0
    a0_max = s / (s - 1)     if s > 1 else 2.0
    P0 = np.ones((B_in, B_out)) / B_out
    alpha0 = np.linspace(a0_min, a0_max, B_out)
    x0 = np.concatenate([P0.flatten(), alpha0])

    # --- Solve ---
    result = optimize.minimize(
        objective, x0,
        method='trust-constr',
        jac=obj_jac,
        hess=obj_hess,
        bounds=bounds,
        constraints=[c_dp, c_row, c_unb],
        options={'verbose': 2 if verbose else 0, 'maxiter': 5000, 'gtol': 1e-8,
                 'sparse_jacobian': True}
    )

    if verbose or not result.success:
        print(f"  Optimizer: {result.message}")

    P = result.x[:n_P].reshape(B_in, B_out)
    alpha = result.x[n_P:]

    # Clean up numerical noise
    P = np.maximum(P, 1e-15)
    P /= P.sum(axis=1, keepdims=True)

    # Sort alpha in ascending order for interpretability
    perm = np.argsort(alpha)
    alpha = alpha[perm]
    P = P[:, perm]

    return P, alpha


def _analytical_budget1(epsilon):
    """Analytical optimal 1-bit LDP mechanism (identical to LDPFL)."""
    s = math.exp(epsilon)
    p = np.array([s / (1 + s), 1.0 / (1 + s)])
    alpha = np.array([-1.0 / (s - 1), s / (s - 1)])
    return p, alpha


def compute_mvu_params_budget1(epsilon, input_bits=3):
    """
    For budget=1 with input_bits>1, build a (B_in, 2) P matrix analytically.
    Each row i uses probability p = (P(z=1|i/(B_in-1))) from the known optimal
    1-bit LDP mechanism (Bernoulli with probability proportional to input value).
    """
    s = math.exp(epsilon)
    B_in = 2 ** input_bits
    target = np.arange(B_in, dtype=float) / (B_in - 1)
    # Optimal unbiased probability: P(output=1|input=t) = (s*t + (1-t)) / (s+1) [no wait]
    # Actually for 1-bit strict-LDP: P(z=1|x) = p_base for x=0, p_base*s for x=1
    # Clipped to [0,1] and normalized to maintain unbiasedness
    # Simple: P(z=1|x) = 1/(1+s) + x*(s-1)/(s+1) [linear interpolation from p[0] to p[1]]
    # alpha[0] = -1/(s-1), alpha[1] = s/(s-1)
    p0 = 1.0 / (1 + s)
    p1 = s / (1 + s)
    P = np.zeros((B_in, 2))
    P[:, 1] = p0 + target * (p1 - p0)    # P(z=1|input[i]) linear in input
    P[:, 0] = 1 - P[:, 1]
    alpha = np.array([-1.0 / (s - 1), s / (s - 1)])
    return P, alpha


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_params(P, alpha, epsilon):
    B_in, B_out = P.shape
    target = np.arange(B_in, dtype=float) / (B_in - 1)

    bias = np.max(np.abs(P @ alpha - target))
    exp_eps = math.exp(epsilon)
    dp_viol = 0.0
    for i in range(B_in):
        for i2 in range(B_in):
            if i == i2: continue
            ratio = P[i, :] / np.maximum(P[i2, :], 1e-300)
            dp_viol = max(dp_viol, float(ratio.max()))
    mse = (P * (target[:, None] - alpha[None, :]) ** 2).sum() / B_in

    print(f"  Max unbiasedness error: {bias:.2e}  (should be ~0)")
    print(f"  Max LDP ratio:          {dp_viol:.4f}  (bound: {exp_eps:.4f})")
    print(f"  MSE:                    {mse:.6f}")
    print(f"  alpha:                  {np.round(alpha, 4)}")
    print(f"  P[0,:]:                 {np.round(P[0,:], 4)}")
    print(f"  P[-1,:]:                {np.round(P[-1,:], 4)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute I-MVU/MVU mechanism parameters for FL experiments"
    )
    parser.add_argument('--epsilon', type=float, nargs='+', default=[1.0, 3.0, 5.0, 7.0],
                        help='Privacy budget values')
    parser.add_argument('--budget', type=int, nargs='+', default=[1, 2, 4],
                        help='Output bits per coordinate')
    parser.add_argument('--input_bits', type=int, default=3,
                        help='Input quantization bits (B_in = 2^input_bits, default: 3 → 8 levels)')
    parser.add_argument('--output_dir', type=str, default='imvu_params',
                        help='Directory to save parameters')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if file already exists')
    parser.add_argument('--verbose', action='store_true',
                        help='Print optimization progress')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for eps in args.epsilon:
        for b in args.budget:
            fname = os.path.join(args.output_dir, f'imvu_eps{eps}_b{b}.pt')
            if os.path.exists(fname) and not args.overwrite:
                print(f"Skipping eps={eps}, budget={b} (file exists; use --overwrite to recompute)")
                continue

            B_in = 2 ** args.input_bits
            B_out = 2 ** b
            print(f"\nComputing MVU params: epsilon={eps}, budget={b} "
                  f"(B_in={B_in}, B_out={B_out})...")

            P, alpha = compute_mvu_params(eps, b, args.input_bits, args.verbose)

            torch.save({
                'P': torch.tensor(P, dtype=torch.float32),          # (B_in, B_out)
                'alpha': torch.tensor(alpha, dtype=torch.float32),  # (B_out,)
                'epsilon': eps,
                'budget': b,
                'input_bits': args.input_bits,
            }, fname)

            print(f"  Saved: {fname}")
            verify_params(P, alpha, eps)

    print("\nDone.")


if __name__ == '__main__':
    main()

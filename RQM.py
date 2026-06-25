"""
Randomized Quantization Mechanism (RQM)  --  Youn, Hu, Ziani, Abernethy (2023),
"Randomized Quantization is All You Need for Differential Privacy in Federated Learning",
Algorithm 2.

This is a faithful, calibration-free implementation. The hyperparameters
(m, q, Delta) are set BY THE CALLER and never tuned inside this module.

----------------------------------------------------------------------------
MECHANISM (single scalar, centered convention)
----------------------------------------------------------------------------
A client holds a scalar  x in [-c, c]  (c = clip radius). RQM releases an
INTEGER LEVEL INDEX  z in {0, 1, ..., m-1}, costing ceil(log2(m)) bits.

  1. Range extension. Widen the output range by Delta on each side:
         Xmax = c + Delta
     and place m EVENLY SPACED levels across [-Xmax, +Xmax]:
         B(i) = -Xmax + 2*i*Xmax/(m-1),     i = 0, 1, ..., m-1
     (spacing = 2*Xmax/(m-1)).  Delta>0 is REQUIRED for non-trivial privacy.

  2. Subsample levels. The two ENDPOINTS B(0) and B(m-1) are ALWAYS kept.
     Each INTERIOR level i in {1,...,m-2} is kept independently with prob q.
     Let the sorted kept indices be  i_1 = 0 < i_2 < ... < i_l = m-1.

  3. Randomized rounding on the kept grid. Because the endpoints are always
     kept and x in [-c,c] subset [B(0), B(m-1)], there is always a kept pair
     (a, b) of consecutive kept levels with  B(a) <= x <= B(b).  Output:
         z = b   with prob  p_up = (x - B(a)) / (B(b) - B(a))
         z = a   otherwise

----------------------------------------------------------------------------
DECODE (server)
----------------------------------------------------------------------------
The decoded VALUE of a level index z is simply B(z). The mechanism is
UNBIASED: E[B(z)] = x exactly, for every realized kept set (randomized
rounding is unbiased and a bracket always exists), hence unconditionally.

For n clients with indices z_1..z_n, the aggregate estimate of the mean is
        mean_hat = (1/n) * sum_i B(z_i)
which equals the paper's Algorithm-1 closed form (with z_sum = sum_i z_i):
        mean_hat = -Xmax + 2 * z_sum * Xmax / (n * (m - 1)).

----------------------------------------------------------------------------
GENERAL PARAMETER (arbitrary center / radius)
----------------------------------------------------------------------------
For a model parameter w_j in [center_j - radius_j, center_j + radius_j],
feed x = w_j - center_j with c = radius_j, then add center_j back on decode.
The wrappers rqm_encode / rqm_decode_value / rqm_decode_mean do this for you.

----------------------------------------------------------------------------
COMMUNICATION
----------------------------------------------------------------------------
m levels  ->  ceil(log2(m)) bits per parameter per client.
(NOTE: this differs from PBM's parameter, which is the number of binomial
TRIALS and yields m+1 levels. Compare baselines by BITS, not by the symbol m.)
"""

from __future__ import annotations
import numpy as np
from itertools import product

__all__ = [
    "rqm_levels", "rqm_bits",
    "rqm_encode", "rqm_decode_value", "rqm_decode_mean",
    "rqm_sample_centered",
    "rqm_output_pmf", "rqm_local_pure_dp_eps", "rqm_local_renyi",
]


# --------------------------------------------------------------------------
# core geometry
# --------------------------------------------------------------------------
def rqm_levels(c: float, Delta: float, m: int) -> np.ndarray:
    """Return the m evenly spaced quantization level VALUES B(0..m-1)."""
    if m < 2:
        raise ValueError("m must be >= 2 (need both endpoints).")
    if Delta <= 0:
        raise ValueError("Delta must be > 0 for non-trivial privacy.")
    Xmax = c + Delta
    return -Xmax + 2.0 * np.arange(m) * Xmax / (m - 1)


def rqm_bits(m: int) -> int:
    """Communication cost in bits per parameter per client."""
    return int(np.ceil(np.log2(m)))


def _bracket(x: float, B: np.ndarray, kept: np.ndarray):
    """Given sorted kept indices, return (a, b, p_up) for randomized rounding of x."""
    Bk = B[kept]
    j = int(np.searchsorted(Bk, x, side="right")) - 1
    j = min(max(j, 0), len(kept) - 2)          # defensive clamp to a valid bracket
    a, b = int(kept[j]), int(kept[j + 1])
    p_up = (x - B[a]) / (B[b] - B[a])
    return a, b, float(p_up)


# --------------------------------------------------------------------------
# encode / decode  (centered core)
# --------------------------------------------------------------------------
def rqm_sample_centered(x: float, c: float, Delta: float, m: int, q: float,
                        rng: np.random.Generator) -> int:
    """One RQM draw for a centered scalar x in [-c, c]. Returns level index z."""
    x = float(np.clip(x, -c, c))
    B = rqm_levels(c, Delta, m)
    interior = np.arange(1, m - 1)
    if interior.size:
        kept = np.concatenate(([0], interior[rng.random(interior.size) < q], [m - 1]))
    else:                                       # m == 2: endpoints only, q unused
        kept = np.array([0, m - 1])
    a, b, p_up = _bracket(x, B, kept)
    return b if rng.random() < p_up else a


# ---- general-parameter wrappers (arbitrary center & radius) ----
def rqm_encode(w: float, center: float, radius: float, Delta: float, m: int, q: float,
               rng: np.random.Generator) -> int:
    """Client side: encode weight w (clipped to [center +/- radius]) -> level index z."""
    return rqm_sample_centered(w - center, radius, Delta, m, q, rng)


def rqm_decode_value(z: int, center: float, radius: float, Delta: float, m: int) -> float:
    """Server side: decode a single level index z -> real value."""
    return center + rqm_levels(radius, Delta, m)[int(z)]


def rqm_decode_mean(z_indices, center: float, radius: float, Delta: float, m: int) -> float:
    """Server side: decode many clients' indices -> unbiased mean estimate."""
    Xmax = radius + Delta
    z_sum = float(np.sum(z_indices))
    n = len(z_indices)
    return center - Xmax + 2.0 * z_sum * Xmax / (n * (m - 1))


# --------------------------------------------------------------------------
# OPTIONAL diagnostics: exact output distribution & privacy of a chosen config.
# These MEASURE the privacy of given (m, q, Delta); they do NOT tune anything.
# Exact enumeration is 2^(m-2) keep-patterns -> use only for small m (<= ~12);
# for large m, estimate the pmf by Monte Carlo instead.
# --------------------------------------------------------------------------
def rqm_output_pmf(x: float, c: float, Delta: float, m: int, q: float) -> np.ndarray:
    """Exact pmf over levels {0..m-1} for centered input x (enumerates interior patterns)."""
    x = float(np.clip(x, -c, c))
    B = rqm_levels(c, Delta, m)
    pmf = np.zeros(m)
    interior = list(range(1, m - 1))
    for pattern in product((0, 1), repeat=len(interior)):
        w, kept = 1.0, [0]
        for bit, idx in zip(pattern, interior):
            if bit:
                kept.append(idx); w *= q
            else:
                w *= (1.0 - q)
        kept.append(m - 1)
        a, b, p_up = _bracket(x, B, np.array(sorted(kept)))
        pmf[b] += w * p_up
        pmf[a] += w * (1.0 - p_up)
    return pmf


def rqm_local_pure_dp_eps(c: float, Delta: float, m: int, q: float) -> float:
    """Local pure-DP epsilon (alpha->inf): worst-case max |log-ratio| at x=+-c."""
    pc = rqm_output_pmf(c, c, Delta, m, q)
    pn = rqm_output_pmf(-c, c, Delta, m, q)
    mask = (pc > 0) & (pn > 0)
    return float(np.max(np.abs(np.log(pc[mask] / pn[mask]))))


def rqm_local_renyi(c: float, Delta: float, m: int, q: float, alpha: float) -> float:
    """Local Renyi divergence D_alpha(P_{Q(c)} || P_{Q(-c)}) for finite alpha > 1."""
    if alpha <= 1:
        raise ValueError("alpha must be > 1.")
    p = rqm_output_pmf(c, c, Delta, m, q)
    qd = rqm_output_pmf(-c, c, Delta, m, q)
    mask = (p > 0) & (qd > 0)
    return float(np.log(np.sum(p[mask] ** alpha / qd[mask] ** (alpha - 1))) / (alpha - 1))


# ==========================================================================
# SELF-TESTS (run `python rqm.py`). All asserts must pass.
# ==========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    c, Delta, m, q = 1.0, 0.8, 5, 0.4

    # 1) levels are sorted, symmetric, span [-(c+Delta), c+Delta]
    B = rqm_levels(c, Delta, m)
    assert np.allclose(B[0], -(c + Delta)) and np.allclose(B[-1], c + Delta)
    assert np.all(np.diff(B) > 0)

    # 2) exact pmf normalizes and the mechanism is UNBIASED: sum_i pmf[i]*B[i] == x
    for x in np.linspace(-c, c, 9):
        pmf = rqm_output_pmf(x, c, Delta, m, q)
        assert abs(pmf.sum() - 1.0) < 1e-12
        assert abs(np.sum(pmf * B) - x) < 1e-12
    print("[ok] pmf normalizes and decode is exactly unbiased")

    # 3) the SAMPLER (encode path) is unbiased in expectation (Monte Carlo)
    x0, T = 0.37, 400_000
    vals = np.array([B[rqm_sample_centered(x0, c, Delta, m, q, rng)] for _ in range(T)])
    assert abs(vals.mean() - x0) < 5e-3, vals.mean()
    print(f"[ok] sampler unbiased: E[B(z)]~{vals.mean():.4f} vs x={x0}")

    # 4) aggregate decoder matches the per-client average decode
    idx = [rqm_sample_centered(x, c, Delta, m, q, rng) for x in rng.uniform(-c, c, 50)]
    mean_a = np.mean([B[z] for z in idx])
    mean_b = rqm_decode_mean(idx, center=0.0, radius=c, Delta=Delta, m=m)
    assert abs(mean_a - mean_b) < 1e-12
    print("[ok] aggregate closed-form decode == mean of per-client decodes")

    # 5) general center/radius wrapper round-trips and stays unbiased
    center, radius = 3.0, 0.5
    w0 = 3.2
    rng2 = np.random.default_rng(1)
    est = np.mean([rqm_decode_value(rqm_encode(w0, center, radius, Delta, m, q, rng2),
                                    center, radius, Delta, m) for _ in range(200_000)])
    assert abs(est - w0) < 5e-3, est
    print(f"[ok] center/radius wrapper unbiased: E[w_hat]~{est:.4f} vs w={w0}")

    # 6) diagnostics run; privacy tightens as Delta grows (more privacy = smaller eps)
    e_small = rqm_local_pure_dp_eps(c, 0.3, m, q)
    e_large = rqm_local_pure_dp_eps(c, 3.0, m, q)
    assert e_large < e_small
    print(f"[ok] local pure-DP eps: Delta=0.3 -> {e_small:.3f},  Delta=3.0 -> {e_large:.3f}")
    print(f"     local Renyi (alpha=2, Delta=0.8): {rqm_local_renyi(c, Delta, m, q, 2.0):.3f}")

    print("\nAll self-tests passed.")
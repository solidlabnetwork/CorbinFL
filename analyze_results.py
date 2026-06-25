#!/usr/bin/env python3
"""
Unified result analyzer for all federated learning experiments.

Scans ./results/ and reports test accuracy (at the round of peak validation
accuracy) for every method / dataset / configuration combination found.

Duplicate resolution (same logical experiment run multiple times):
  - Keep the file with the most completed rounds.
  - Tie-break: keep the file from the most recent timestamped directory.
"""

import os
import re
import csv
import datetime
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
EXPECTED_SEEDS = [0, 42, 100, 1234, 5678]

# ---------------------------------------------------------------------------
# Filename pattern — covers all method formats produced by main.py:
#
#  {method}_{dataset}_{IID|non-IID}_N_{n}_T_{t}_
#      [ep_{eps}_]
#      [b{n}_]               <- IMVU
#      [m{n}_[q{q}_]]        <- PBM_PLDP / RQM
#      [NR_{n}_]             <- CorbinFL / LDPFL / AugCorbinFL
#  lmbda_{l}_seed_{s}_lr_{lr}.csv
# ---------------------------------------------------------------------------
FILE_PATTERN = re.compile(
    r"^(?P<method>.+?)_"
    r"(?P<dataset>MNIST|CIFAR10|FEMNIST|Shakespeare|Sent140)_"
    r"(?P<iid>IID|non-IID)_"
    r"N_(?P<n_clients>\d+)_"
    r"T_(?P<n_rounds>\d+)_"
    r"(?:ep_(?P<eps>[0-9.]+)_)?"
    r"(?:b(?P<imvu_b>\d+)_)?"
    r"(?:m(?P<m>\d+)_(?:q(?P<q>[0-9.]+)_)?)?"
    r"(?:NR_(?P<num_rand>\d+)_)?"
    r"lmbda_(?P<lmbda>[0-9.]+)_"
    r"seed_(?P<seed>\d+)_"
    r"lr_(?P<lr>[0-9.e+\-]+)\.csv$"
)

# Result directories are named: {method}_{dataset}_{YYYYMMDD}_{HHMMSS}
DIR_TS_RE = re.compile(r"_(\d{8}_\d{6})$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_dir_timestamp(dir_name: str) -> datetime.datetime:
    m = DIR_TS_RE.search(dir_name)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.datetime.min


def make_budget_str(m):
    parts = []
    if m.group('imvu_b') is not None:
        parts.append(f"b={m.group('imvu_b')}")
    if m.group('m') is not None:
        if m.group('q') is not None:
            parts.append(f"m={m.group('m')}, q={m.group('q')}")
        else:
            parts.append(f"m={m.group('m')}")
    if m.group('num_rand') is not None:
        parts.append(f"NR={m.group('num_rand')}")
    return ', '.join(parts) if parts else None


def count_csv_rows(filepath: str) -> int:
    """Count data rows (excluding header) in a CSV file."""
    try:
        with open(filepath, newline='') as f:
            return sum(1 for line in f if line.strip()) - 1
    except Exception:
        return 0


def read_best_val_test_acc(filepath: str):
    """Return test accuracy at the round with the highest validation accuracy."""
    best_val  = -1.0
    best_test = None
    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_acc = float(row['Val Accuracy'])
                if val_acc > best_val:
                    best_val  = val_acc
                    best_test = float(row['Test Accuracy'])
        return best_test
    except Exception as e:
        print(f"  [WARN] Could not read {os.path.basename(filepath)}: {e}")
        return None


# ---------------------------------------------------------------------------
# Discovery — collect candidate files for every logical experiment key
#
# Logical key:
#   (method, dataset, iid, n_clients, n_rounds, budget_str, lr, eps, lmbda, seed)
# ---------------------------------------------------------------------------
candidates: dict = defaultdict(list)   # key -> [(filepath, row_count, timestamp)]

for entry in os.scandir(RESULTS_DIR):
    if not entry.is_dir():
        continue
    ts = parse_dir_timestamp(entry.name)
    for f in os.scandir(entry.path):
        if not f.name.endswith('.csv'):
            continue
        m = FILE_PATTERN.match(f.name)
        if not m:
            continue
        eps_raw = m.group('eps')
        key = (
            m.group('method'),
            m.group('dataset'),
            m.group('iid'),
            int(m.group('n_clients')),
            int(m.group('n_rounds')),
            make_budget_str(m),
            float(m.group('lr')),
            float(eps_raw) if eps_raw else None,
            round(float(m.group('lmbda')), 2),
            int(m.group('seed')),
        )
        candidates[key].append((f.path, count_csv_rows(f.path), ts))

# ---------------------------------------------------------------------------
# Resolve duplicates
#   Winner = most rows completed; tie-break = most recent directory timestamp
# ---------------------------------------------------------------------------
n_duplicates = sum(len(v) - 1 for v in candidates.values() if len(v) > 1)
if n_duplicates:
    print(f"[INFO] Resolved {n_duplicates} duplicate file(s) "
          f"(kept most-rounds, then most-recent).\n")

results: dict = {}
for key, file_list in candidates.items():
    winner_path, _, _ = max(file_list, key=lambda x: (x[1], x[2]))
    acc = read_best_val_test_acc(winner_path)
    if acc is not None:
        results[key] = acc

# ---------------------------------------------------------------------------
# Organise into display structure
#
#   section key : (method, dataset, iid, n_clients, n_rounds, budget_str, lr)
#   data[section][eps][lmbda][seed] = test_acc
# ---------------------------------------------------------------------------
data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for (method, dataset, iid, n_clients, n_rounds, budget, lr,
        eps, lmbda, seed), acc in results.items():
    sec = (method, dataset, iid, n_clients, n_rounds, budget, lr)
    data[sec][eps][lmbda][seed] = acc


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def sorted_eps(eps_keys):
    return sorted(eps_keys, key=lambda e: float('inf') if e is None else e)


def print_lambda_table(eps_data: dict, indent: str = '        ') -> list:
    """
    Print the lambda table for one (section, eps) block.
    Returns a list of (lmbda, missing_seeds) for incomplete rows.
    """
    lmbdas = sorted(eps_data.keys())

    # Best lambda = highest average test accuracy over available seeds
    best_lmbda, best_avg = None, -1.0
    for lmbda, seed_map in eps_data.items():
        accs = list(seed_map.values())
        if accs and np.mean(accs) > best_avg:
            best_avg  = np.mean(accs)
            best_lmbda = lmbda

    hdr = (f"{indent}{'':2} {'Lambda':>8} | {'Avg Acc':>8} | "
           f"{'Std':>8} | {'N':>2}/{len(EXPECTED_SEEDS)} | Seeds present")
    sep = f"{indent}{'-' * 64}"
    print(hdr)
    print(sep)

    missing_rows = []
    for lmbda in lmbdas:
        seed_map = eps_data[lmbda]
        present  = sorted(seed_map.keys())
        missing  = [s for s in EXPECTED_SEEDS if s not in seed_map]
        accs     = [seed_map[s] for s in present]
        marker   = '★ ' if lmbda == best_lmbda else '  '

        if accs:
            avg_s = f"{np.mean(accs):8.2f}"
            std_s = f"{np.std(accs):8.4f}"
        else:
            avg_s = std_s = "     N/A"

        miss_s = f"  [missing: {missing}]" if missing else ""
        print(f"{indent}{marker}{lmbda:8.2f} | {avg_s} | {std_s} | "
              f"{len(accs):>2}/{len(EXPECTED_SEEDS)} | {present}{miss_s}")

        if missing:
            missing_rows.append((lmbda, missing))

    return missing_rows


def section_sort_key(k):
    method, dataset, iid, n_clients, n_rounds, budget, lr = k
    # eps-less methods (budget=None) sort after methods with budgets
    return (method, dataset, iid, n_clients, n_rounds, budget or '', lr)


# ---------------------------------------------------------------------------
# Main display loop
# ---------------------------------------------------------------------------
all_missing = []   # (method, dataset, iid, N, T, budget, lr, eps, lmbda, missing_seeds)

prev_method    = None
prev_ds_config = None
prev_budget_lr = None

for sec in sorted(data.keys(), key=section_sort_key):
    method, dataset, iid, n_clients, n_rounds, budget, lr = sec

    # ---- Method header ----
    if method != prev_method:
        print(f"\n{'=' * 80}")
        print(f"  Method: {method}")
        print(f"{'=' * 80}")
        prev_method    = method
        prev_ds_config = None
        prev_budget_lr = None

    # ---- Dataset / experiment-config header ----
    ds_config = (dataset, iid, n_clients, n_rounds)
    if ds_config != prev_ds_config:
        print(f"\n  Dataset: {dataset}  |  {iid}  |  N={n_clients}, T={n_rounds}")
        print(f"  {'-' * 70}")
        prev_ds_config = ds_config
        prev_budget_lr = None

    # ---- Budget / lr sub-header ----
    budget_lr = (budget, lr)
    if budget_lr != prev_budget_lr:
        label_parts = ([budget] if budget else []) + [f"lr={lr}"]
        print(f"\n    [ {' | '.join(label_parts)} ]")
        prev_budget_lr = budget_lr

    # ---- Epsilon blocks ----
    for eps in sorted_eps(data[sec].keys()):
        lmbda_data = data[sec][eps]

        if eps is not None:
            print(f"\n      epsilon = {eps}")
        else:
            print(f"\n      (no privacy budget)")

        missing_rows = print_lambda_table(lmbda_data)
        for lmbda, miss in missing_rows:
            all_missing.append(
                (method, dataset, iid, n_clients, n_rounds, budget, lr, eps, lmbda, miss)
            )

# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"  SUMMARY")
print(f"{'=' * 80}")
print(f"  Total result files loaded : {len(results)}")
print(f"  Unique experiment keys    : {len(candidates)}")
if n_duplicates:
    print(f"  Duplicates resolved       : {n_duplicates}")

if all_missing:
    print(f"\n  Incomplete seed sets ({len(all_missing)} lambda-row(s) missing at least one seed):")
    for (method, dataset, iid, N, T, budget, lr, eps, lmbda, miss) in all_missing:
        eps_s = f"eps={eps}" if eps is not None else "no-eps"
        bud_s = f", {budget}" if budget else ""
        print(f"    {method} | {dataset} | {iid} | N={N}, T={T}{bud_s} | "
              f"{eps_s} | lmbda={lmbda:.2f}  ->  missing seeds {miss}")
else:
    print(f"\n  All expected seeds present for every combination found.")

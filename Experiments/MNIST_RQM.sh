#!/bin/bash
# RQM experiments on MNIST (IID).
# m=2: 1 bit per coordinate, q unused (no interior levels). Single q=0.5 placeholder.
# m=3: 2 bits per coordinate, q swept 0.1-0.9; best q reported.
# Delta is auto-calibrated from epsilon at construction — no manual tuning needed.
cd ..

seeds=(0 42 100 1234 5678)
epsilons=(0.1 0.3 1 3)
qs=(0.3 0.4 0.5 0.6)
lambda_params=($(seq 0.1 0.1 1.0))

run_with_retry() {
    local cmd="$1"
    local retries=3
    local delay=600
    for ((i=1; i<=retries; i++)); do
        echo "Attempt $i: $cmd"
        eval "$cmd"
        if [ $? -eq 0 ]; then return 0; fi
        echo "Failed attempt $i"
        if [ $i -lt $retries ]; then sleep $delay; fi
    done
    echo "Command failed after $retries attempts: $cmd"
    return 1
}

# --- m=2 (1-bit): q has no effect, run once per (seed, epsilon, lambda) ---
for seed_val in "${seeds[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        for lambda in "${lambda_params[@]}"; do
            cmd="python main.py --dataset MNIST --iid --method RQM --device GPU \
                --num_clients 50 --num_rounds 100 --epsilon $epsilon \
                --rqm_m 2 --rqm_q 0.5 --lambda_param $lambda \
                --seed $seed_val --lr 0.001 --weight_decay 0.003 --rqm_r_max 3"
            run_with_retry "$cmd"
        done
    done
done

# # --- m=3 (2-bit): sweep q to find best ---
# for seed_val in "${seeds[@]}"; do
#     for epsilon in "${epsilons[@]}"; do
#         for q in "${qs[@]}"; do
#             for lambda in "${lambda_params[@]}"; do
#                 cmd="python main.py --dataset MNIST --iid --method RQM --device GPU \
#                     --num_clients 50 --num_rounds 100 --epsilon $epsilon \
#                     --rqm_m 3 --rqm_q $q --lambda_param $lambda \
#                     --seed $seed_val --lr 0.001 --weight_decay 0.003"
#                 run_with_retry "$cmd"
#             done
#         done
#     done
# done

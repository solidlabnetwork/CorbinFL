#!/bin/bash
# PBM-PLDP experiments on MNIST (IID).
# m=1: 1 bit per coordinate,  theta = 0.5*tanh(eps/2)
# m=2: 2 bits per coordinate, theta = 0.5*tanh(eps/4), lower variance at 2x comm cost.
cd ..

seeds=(0 42 100 1234 5678)
epsilons=(0.1 0.3 1.0 3.0)
ms=(1 2)
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

for seed_val in "${seeds[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        for m in "${ms[@]}"; do
            for lambda in "${lambda_params[@]}"; do
                cmd="python main.py --dataset MNIST --iid --method PBM_PLDP --device GPU \
                    --num_clients 50 --num_rounds 100 --epsilon $epsilon \
                    --pbm_m $m --lambda_param $lambda \
                    --pbm_r_max 3.0 \
                    --seed $seed_val --lr 0.001 --weight_decay 0.003"
                run_with_retry "$cmd"
            done
        done
    done
done

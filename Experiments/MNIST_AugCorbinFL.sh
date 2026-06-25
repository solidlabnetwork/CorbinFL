#!/bin/bash
# AugCorbinFL experiments on MNIST (IID), gamma=0.2.
cd ..

seeds=(0 42 100 1234 5678)
epsilons=(0.1 0.5 1 3)
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
        for lambda in "${lambda_params[@]}"; do
            cmd="python main.py --dataset MNIST --iid --method AugCorbinFL --device GPU \
                --num_clients 50 --num_rounds 100 --epsilon $epsilon \
                --num_rand 5 --Gamma 0.2 --lambda_param $lambda \
                --seed $seed_val --lr 0.001 --weight_decay 0.003"
            run_with_retry "$cmd"
        done
    done
done

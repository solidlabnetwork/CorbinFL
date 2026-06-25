#!/bin/bash
# CorQuant experiments on CIFAR-10 (IID).
# CorQuant is a correlated quantization scheme — no privacy budget (epsilon-free).
cd ..

seeds=(0 42 100 1234 5678)
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
    for lambda in "${lambda_params[@]}"; do
        cmd="python main.py --dataset CIFAR10 --iid --method CorQuant --device GPU \
            --num_clients 50 --num_rounds 100 \
            --lambda_param $lambda \
            --seed $seed_val --lr 0.001 --weight_decay 0.003"
        run_with_retry "$cmd"
    done
done

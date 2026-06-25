#!/bin/bash
# SignSGD (majority vote) experiments on MNIST (IID).
# Non-private baseline: 1 bit/coordinate, no differential privacy.
cd ..

seeds=(0 42 100 1234 5678)
lrs=(0.0001 0.0003 0.001)

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
    for lr in "${lrs[@]}"; do
        cmd="python main.py --dataset MNIST --iid --method SignSGD --device GPU \
            --num_clients 50 --num_rounds 100 \
            --seed $seed_val --lr $lr --weight_decay 0"
        run_with_retry "$cmd"
    done
done

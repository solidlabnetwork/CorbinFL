#!/bin/bash
# Move to the parent directory
cd ..

# Define parameters
seeds=(0 42 100 1234 5678)
epsilons=(1 5)
# Generate lambda parameters from 0.1 to 1.0 with step 0.1
lambda_params=($(seq 0.1 0.1 1.0))

# Retry function
run_with_retry() {
    local cmd="$1"
    local retries=3
    local delay=600

    for ((i=1; i<=retries; i++)); do
        echo "Attempt $i: $cmd"
        eval "$cmd"
        if [ $? -eq 0 ]; then
            echo "Command succeeded on attempt $i"
            return 0
        else
            echo "Command failed on attempt $i"
            if [ $i -lt $retries ]; then
                echo "Retrying in $delay seconds..."
                sleep $delay
            fi
        fi
    done

    echo "Command failed after $retries attempts: $cmd"
    return 1
}

# Loop through all combinations of parameters
for seed_val in "${seeds[@]}"; do
    echo "Running experiments with seed: $seed_val"
    
    for epsilon in "${epsilons[@]}"; do
        for lambda in "${lambda_params[@]}"; do
            cmd="python main.py --dataset Shakespeare --iid --method LaplaceLDP --device GPU --num_clients 150 --num_rounds 200 --epsilon $epsilon --lambda_param $lambda --seed $seed_val --lr 0.001"
            run_with_retry "$cmd"
        done
    done
done
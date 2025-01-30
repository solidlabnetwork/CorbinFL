
#!/bin/bash
# Move to the parent directory
cd ..
# Define seeds as an array
# seeds = [0, 42, 100, 1234, 5678, 9876, 13579, 24680, 31415, 99999]
seeds=(0 42 100 1234 5678)

# Retry function
run_with_retry() {
    local cmd="$1"
    local retries=3          # Number of retries
    local delay=600          # Delay between retries (in seconds)

    for ((i=1; i<=retries; i++)); do
        echo "Attempt $i: $cmd"
        eval "$cmd"          # Execute the command
        if [ $? -eq 0 ]; then
            echo "Command succeeded on attempt $i"
            return 0         # Exit the function if command succeeds
        else
            echo "Command failed on attempt $i"
            if [ $i -lt $retries ]; then
                echo "Retrying in $delay seconds..."
                sleep $delay
            fi
        fi
    done

    echo "Command failed after $retries attempts: $cmd"
    return 1                 # Return failure if all retries fail
}

# Loop through each seed value
for seed_val in "${seeds[@]}"; do
    echo "Running experiments with seed: $seed_val"

    # Experiment commands with retry logic
    run_with_retry "python main.py --dataset CIFAR10 --iid --method FedAvg --device GPU --num_clients 50 --num_rounds 100 --seed $seed_val --lr 0.001 --weight_decay 0.003"


done

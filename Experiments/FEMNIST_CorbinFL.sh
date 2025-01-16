
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
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 10 --num_rand 5 --lambda_param 1 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 10 --num_rand 5 --lambda_param 0.9 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 5 --num_rand 5 --lambda_param 1 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 5 --num_rand 5 --lambda_param 0.9 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 5 --num_rand 5 --lambda_param 0.8 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 5 --num_rand 5 --lambda_param 0.7 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 1 --num_rand 5 --lambda_param 0.1 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 1 --num_rand 5 --lambda_param 0.2 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 1 --num_rand 5 --lambda_param 0.3 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 1 --num_rand 5 --lambda_param 0.4 --seed $seed_val --lr 0.0003"
    run_with_retry "python main.py --dataset FEMNIST --iid --method CorbinFL --device GPU --num_clients 50 --num_rounds 100 --epsilon 1 --num_rand 5 --lambda_param 0.5 --seed $seed_val --lr 0.0003"

done

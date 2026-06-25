#!/bin/bash
# I-MVU experiments on CIFAR-10 (IID)
# budget=1: 1-bit output, achieves same MSE as LDPFL (optimizer converges to the same solution).
# budget=2: 2-bit output, lower MSE at 2x communication cost vs CorBinFL.
# beta=1.0: paper's recommended value for MNIST/CIFAR10 (Section 3.1).
cd ..

seeds=(0 42 100 1234 5678)
epsilons=(1 3 5 7)
budgets=(1)
lambda_params=($(seq 0.1 0.1 1.0))

# Precompute I-MVU params first (skips existing files automatically)
python precompute_imvu.py --epsilon "${epsilons[@]}" --budget "${budgets[@]}"

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
        for budget in "${budgets[@]}"; do
            for lambda in "${lambda_params[@]}"; do
                cmd="python main.py --dataset CIFAR10 --iid --method IMVU --device GPU \
                    --num_clients 50 --num_rounds 100 --epsilon $epsilon \
                    --imvu_budget $budget --imvu_beta 1.0 --lambda_param $lambda \
                    --seed $seed_val --lr 0.001 --weight_decay 0.003"
                run_with_retry "$cmd"
            done
        done
    done
done

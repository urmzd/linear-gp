#!/usr/bin/env bash

# Define the array of environments
environments=("mountain-car-q" "mountain-car-lgp" "iris" "cart-pole-q" "cart-pole-lgp")

# Set the number of trials and threads
n_trials=40
n_threads=4

# Iterate through the environments and run the search.py script
for env in "${environments[@]}"; do
    echo "Running search.py for environment: $env"
    ./scripts/search.py --env "$env" --n-trials "$n_trials" --n-threads "$n_threads" &
done

echo "All variations completed."

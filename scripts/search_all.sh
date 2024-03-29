#!/usr/bin/env bash

# Define the arrays of environments
lgp_environments=(
    "cart-pole-lgp"
    "mountain-car-lgp"
)

q_environments=(
    "mountain-car-q"
    "cart-pole-q"
)

# Function to run the search.py script for each environment
run_search() {
    local environments=("$@")
    for env in "${environments[@]}"; do
        echo "Running search.py for environment: $env"
        ./scripts/search.py --env "$env" --n-trials "$SEARCH_ALL_N_TRIALS" --n-threads "$SEARCH_ALL_N_THREADS" &
    done
    wait  # Wait for all background processes to complete
}

# Run the search.py script for lgp environments
echo "Running LGP environments..."
run_search "${lgp_environments[@]}"

# Run the search.py script for q environments
echo "Running Q environments..."
run_search "${q_environments[@]}"

echo "All variations completed."

#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <steps> <batch_size>"
    exit 1
fi

# Iterate from 0 to 8 for model_id
for model_id in {0..2}; do
    echo "Running main.py with model_id=${model_id}, steps=$1, batch_size=$2"
    python run_exp_single_model.py ${model_id} $1 $2
done

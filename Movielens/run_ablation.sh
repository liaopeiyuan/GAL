#!/bin/env bash

for f in ./config/Node_Attack/ablation/*.json; do
    echo "Running $f..."
    python exec.py --config_path=$f
    trap - SIGTSTP
done

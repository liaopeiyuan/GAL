#!/bin/env bash

for f in ./config/Node_Attack/paper/*.json; do
    echo "Running $f..."
    CUDA_VISIBLE_DEVICES=0 python exec.py --config_path=$f
    trap - SIGTSTP
done

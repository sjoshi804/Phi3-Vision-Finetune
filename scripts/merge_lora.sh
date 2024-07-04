#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /path/to/model \
    --model-base microsoft/Phi-3-vision-128k-instruct  \
    --save-model-path /path/to/save \
    --safe-serialization
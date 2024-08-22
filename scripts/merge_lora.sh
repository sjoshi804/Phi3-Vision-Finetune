#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

export PYTHONPATH=src:$PYTHONPATH

# You should add load-8bit or load-4bit if you've trained with QLoRA.
python src/merge_lora_weights.py \
    --model-path /path/to/model \
    --model-base $MODEL_NAME  \
    --save-model-path /path/to/save \
    --safe-serialization
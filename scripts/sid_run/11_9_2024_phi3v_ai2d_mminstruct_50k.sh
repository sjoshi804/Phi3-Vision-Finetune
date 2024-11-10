#!/bin/bash
export PYTHONPATH=src:$PYTHONPATH

#!/bin/bash

TRAIN_DATA_PATH=/home/sjoshi/llava-fork/mminstruct_data/ai2d_gen_task_desc_50k_middle_ppl.json
RUN_ID=ai2d_gen_task_desc_50k_middle_ppl

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id microsoft/Phi-3-vision-128k-instruct \
    --data_path $TRAIN_DATA_PATH \
    --image_folder /path/to/your/image/folder \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir checkpoints/phi3v_$RUN_ID \
    --num_crops 16 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --projector_lr 2e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --dataloader_num_workers 4
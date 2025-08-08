#!/bin/bash


export PYTHONPATH=$PYTHONPATH:your/path/to/grok


# 语言模型基座 (Qwen2)
LLM_VERSION="your/path/to/Qwen2-7B-Instruct"

# 视觉编码器权重路径
CFP_TOWER_PATH="your/path/to/CFP_encoder.pth"
OCT_TOWER_PATH="your/path/to/OCT_encoder.pth"

# 数据路径
DATA_PATH="your/path/to/single_disease_train_data_.json"
IMAGE_FOLDER="your/path/to/CFP_data" # CFP图像从此读取
OCT_FOLDER="your/path/to/OCT_data"   # OCT图像从此读取

# 训练产物的输出目录
OUTPUT_DIR="./checkpoints/llava-qwen2-lora-$(date +%Y%m%d)" # 加入日期，方便区分实验

# =========================================================================
# 2. DeepSpeed 启动命令 (参数已优化)####################
# =========================================================================
deepspeed --num_gpus=1 --master_port 29506 your/path/to/grok/llava/train/train.py \
    --deepspeed ./zero2.json \
    \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --tune_mm_mlp_adapter True \
    \
    --model_name_or_path ${LLM_VERSION} \
    --version qwen_2 \
    \
    --vision_tower ${CFP_TOWER_PATH} \
    --mm_projector_type mlp2x_gelu \
    \
    --oct_2d_tower ${OCT_TOWER_PATH} \
    --oct_2d_projector_type mlp2x_gelu \
    \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --oct_folder ${OCT_FOLDER} \
    --image_aspect_ratio pad \
    --lazy_preprocess True \
    --use_weighted_sampling False \
    \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to "tensorboard" 
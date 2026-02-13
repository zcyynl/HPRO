#!/bin/bash

# 示例训练脚本
# 根据您的硬件和数据调整参数

NUM_GPUS=8
DATA_FILE="data/train_data.pkl"
OUTPUT_DIR="output/experiment_1"
PRETRAINED_MODEL="/path/to/Qwen1.5-1.8B"
DEEPSPEED_CONFIG="cfg/ds_config_bf16_stage2.json"

BATCH_SIZE=32
EPOCHS=10
LEARNING_RATE=1e-4
WARMUP_STEPS=4000
MAX_LEN=3600
LORA_R=16
LORA_ALPHA=1.0

mkdir -p ${OUTPUT_DIR}

# 选项1: 标准训练（不使用HPRO）
echo "Starting training without HPRO..."
deepspeed --num_gpus=${NUM_GPUS} src/train.py \
    --deepspeed_config ${DEEPSPEED_CONFIG} \
    --data_file ${DATA_FILE} \
    --pretrained_model ${PRETRAINED_MODEL} \
    --out_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --warmup_step_num ${WARMUP_STEPS} \
    --max_len ${MAX_LEN} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --k 128 \
    --dropout_rate 0.5 \
    --use_pairwise \
    --pairwise_margin 0.0 \
    --ckpt_interval 10240 \
    --pr_threshold 0.5 \
    --data_aug_n 0 \
    2>&1 | tee ${OUTPUT_DIR}/train.log

# 选项2: 使用HPRO训练
# 取消下面的注释来启用HPRO
# echo "Starting training with HPRO..."
# deepspeed --num_gpus=${NUM_GPUS} src/train.py \
#     --deepspeed_config ${DEEPSPEED_CONFIG} \
#     --data_file ${DATA_FILE} \
#     --pretrained_model ${PRETRAINED_MODEL} \
#     --out_dir ${OUTPUT_DIR}_hpro \
#     --batch_size ${BATCH_SIZE} \
#     --epochs ${EPOCHS} \
#     --lr ${LEARNING_RATE} \
#     --warmup_step_num ${WARMUP_STEPS} \
#     --max_len ${MAX_LEN} \
#     --lora_r ${LORA_R} \
#     --lora_alpha ${LORA_ALPHA} \
#     --k 128 \
#     --dropout_rate 0.5 \
#     --use_hpro \
#     --hpro_margin_global 1.0 \
#     --hpro_margin_key 0.5 \
#     --hpro_margin_soft 0.1 \
#     --ckpt_interval 10240 \
#     --pr_threshold 0.5 \
#     --data_aug_n 0 \
#     2>&1 | tee ${OUTPUT_DIR}_hpro/train.log

echo "训练完成。Checkpoints保存在 ${OUTPUT_DIR}"

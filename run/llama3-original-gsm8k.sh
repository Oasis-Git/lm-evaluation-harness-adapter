#!/bin/bash
#SBATCH --job-name=llama3-original-gsm8k
#SBATCH --output=/home/yuweia/lm-evaluation-harness-adapter/run/log/llama3-original-gsm8k.log
#SBATCH --error=/home/yuweia/lm-evaluation-harness-adapter/run/log/llama3-original-gsm8k.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2:00:00
#SBATCH --partition=general

export TRANSFORMERS_CACHE=/data/user_data/yuweia/HuggingFace_Models
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks gsm8k_cot \
    --device cuda:0 \
    --batch_size 1 \
    --limit 200

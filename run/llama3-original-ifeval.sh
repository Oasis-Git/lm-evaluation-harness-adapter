#!/bin/bash
#SBATCH --job-name=llama3-original-ifeval
#SBATCH --output=/home/yuweia/lm-evaluation-harness-adapter/run/log/llama3-original-ifeval.log
#SBATCH --error=/home/yuweia/lm-evaluation-harness-adapter/run/log/llama3-original-ifeval.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2:00:00
#SBATCH --partition=general

export TRANSFORMERS_CACHE=/data/user_data/yuweia/HuggingFace_Models
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True \
    --tasks ifeval \
    --device cuda:0 \
    --batch_size 1 \
    --limit 3

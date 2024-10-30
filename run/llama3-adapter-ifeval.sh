#!/bin/bash
#SBATCH --job-name=llama3-adapter-ifeval
#SBATCH --output=/home/yuweia/lm-evaluation-harness-adapter/run/log/llama3-adapter-ifeval.log
#SBATCH --error=/home/yuweia/lm-evaluation-harness-adapter/run/log/llama3-adapter-ifeval.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2:00:00
#SBATCH --partition=general

export TRANSFORMERS_CACHE=/data/user_data/yuweia/HuggingFace_Models
cd /home/yuweia/lm-evaluation-harness-adapter
python llama_infer.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --request_file_path /home/yuweia/lm-evaluation-harness-adapter/requests/ifeval.jsonl \
    --device cuda:0 \
    --batchsize 4 \
    --response_file_path /home/yuweia/lm-evaluation-harness-adapter/responses/ifeval.jsonl \
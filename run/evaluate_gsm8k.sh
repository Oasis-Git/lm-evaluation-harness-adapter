#!/bin/bash
#SBATCH --job-name=evaluate_gsm8k
#SBATCH --output=/home/yuweia/lm-evaluation-harness-adapter/run/log/evaluate_gsm8k.log
#SBATCH --error=/home/yuweia/lm-evaluation-harness-adapter/run/log/evaluate_gsm8k.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=2:00:00
#SBATCH --partition=general

export TRANSFORMERS_CACHE=/data/user_data/yuweia/HuggingFace_Models
cd /home/yuweia/lm-evaluation-harness-adapter
python evaluate_response.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --response_file_path /home/yuweia/lm-evaluation-harness-adapter/responses/gsm8k_cot.jsonl \
    --task_name gsm8k_cot \
    --limit 200
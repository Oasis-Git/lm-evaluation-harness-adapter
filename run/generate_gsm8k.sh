#!/bin/bash
#SBATCH --job-name=generate_gsm8k
#SBATCH --output=/home/yuweia/lm-evaluation-harness-adapter/run/log/generate_gsm8k.log
#SBATCH --error=/home/yuweia/lm-evaluation-harness-adapter/run/log/generate_gsm8k.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=2:00:00
#SBATCH --partition=general

export TRANSFORMERS_CACHE=/data/user_data/yuweia/HuggingFace_Models
cd /home/yuweia/lm-evaluation-harness-adapter
python generate_request.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --request_file_path /home/yuweia/lm-evaluation-harness-adapter/requests/gsm8k_cot.jsonl \
    --task_name gsm8k_cot

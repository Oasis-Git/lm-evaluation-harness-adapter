import json
import os
import argparse

from lm_eval.models.evaluate_lm import EvaluateLM
from lm_eval import evaluator, tasks
from lm_eval import evaluator, tasks
from lm_eval.processor import read_responses_processor
import lm_eval.api as lmeval
from lm_eval.api.registry import ALL_TASKS
import lm_eval
from lm_eval.utils import make_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--response_file_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--num_fewshot", type=int, default=0)
    args = parser.parse_args()

    processor = read_responses_processor(args.response_file_path)
    model = EvaluateLM(pretrained=args.model,
                       processor=processor,
                       batch_size=args.batchsize)

    task_manager = lm_eval.tasks.TaskManager("INFO")
    task_list = [args.task_name]
    task_names = task_manager.match_tasks(task_list)

    results = evaluator.simple_evaluate(model,
                                        None,
                                        tasks=task_names,
                                        batch_size=args.batchsize,
                                        limit=args.limit,
                                        num_fewshot=args.num_fewshot)
    print(make_table(results))

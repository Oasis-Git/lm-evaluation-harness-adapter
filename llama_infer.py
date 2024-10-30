from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
import os
import torch
import json
from tqdm import tqdm
import argparse
import transformers
from typing import List
from lm_eval.models.utils import MultiTokenEOSCriteria


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--request_file_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--response_file_path", type=str, required=True)

    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # read all the requests
    requests = []
    with open(args.request_file_path, "r") as f:
        for line in f:
            requests.append(json.loads(line))

    # generate responses with batchsize
    responses = []
    for i in tqdm(range(0, len(requests), args.batchsize)):
        # create context
        context = [request["context"] for request in requests[i : i + args.batchsize]]
        until = requests[i]["until"]
        # create kwargs with left over keys
        kwargs = {}
        for request in requests[i : i + args.batchsize]:
            kwargs.update(
                {k: v for k, v in request.items() if k not in ["context", "until"]}
            )

        # tokenize the context
        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        stopping_criteria = stop_sequences_criteria(
            tokenizer, until, inputs["input_ids"].shape[1], inputs["input_ids"].shape[0]
        )
        output = model.generate(**inputs, stopping_criteria=stopping_criteria, **kwargs)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        responses.extend(output)

    # write the responses
    with open(args.response_file_path, "w+") as f:
        for i in range(len(requests)):
            d = dict()
            d["request"] = requests[i]
            d["response"] = responses[i]
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()

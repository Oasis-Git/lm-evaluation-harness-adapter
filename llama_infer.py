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
from typing import List, Union
from lm_eval.models.utils import MultiTokenEOSCriteria


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList([
        *[
            MultiTokenEOSCriteria(sequence, tokenizer,
                                  initial_decoder_input_length, batch_size)
            for sequence in stop_sequences
        ],
    ])


def configure_pad_token(tokenizer, ):
    """
    This function checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.
    Some tokenizers require special handling.

    Args:
        tokenizer: The tokenizer for which the padding token is to be handled.
        model_config: The configuration of the model. Default is None.

    Returns:
        The tokenizer after the padding token has been handled.

    Raises:
        AssertionError: If the tokenizer is of type RWKVWorldTokenizer or Rwkv5Tokenizer and the padding token id is not 0.
    """
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print("No padding token found. Setting padding token to eos token.")
        exit(1)


def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--request_file_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--response_file_path", type=str, required=True)

    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 trust_remote_code=True,
                                                 torch_dtype='auto')
    print(next(model.parameters()).dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    configure_pad_token(tokenizer)
    tokenizer.padding_side = 'left'
    model.to(args.device)
    model.eval()

    special_tokens_kwargs = {"add_special_tokens": False}

    # read all the requests
    requests = []
    with open(args.request_file_path, "r") as f:
        for line in f:
            requests.append(json.loads(line))

    # generate responses with batchsize
    responses = []
    for i in tqdm(range(0, len(requests), args.batchsize)):
        # create context
        context = [
            request["context"] for request in requests[i:i + args.batchsize]
        ]
        until = requests[i]["until"]
        # create kwargs with left over keys
        kwargs = {}
        for request in requests[i:i + args.batchsize]:
            kwargs.update({
                k: v
                for k, v in request.items() if k not in ["context", "until"]
            })

        # tokenize the context
        inputs = tokenizer(context,
                           truncation=False,
                           padding="longest",
                           return_tensors="pt",
                           **special_tokens_kwargs)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        if kwargs.get("do_sample", False) == False:
            # delete the top_p and top_k arguments
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)
            kwargs.pop("temperature", None)
        stopping_criteria = stop_sequences_criteria(
            tokenizer, until, inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0])
        if kwargs.get("max_new_tokens", None) is not None:
            kwargs["max_length"] = inputs["input_ids"].shape[1] + kwargs[
                "max_new_tokens"]
            kwargs.pop("max_new_tokens", None)

        output = model.generate(**inputs,
                                stopping_criteria=stopping_criteria,
                                pad_token_id=tokenizer.pad_token_id,
                                **kwargs)
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

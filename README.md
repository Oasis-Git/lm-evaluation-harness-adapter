# Language Model Evaluation Harness Adapter

This repository is a fork of [lm-eval](https://github.com/eleutherai/lm-eval), an evaluation framework for large language models (LLMs). The primary goal of **LM Eval Adapter** is to enhance flexibility and usability, enabling multi-LLM evaluation and supporting approximate LLM inference techniques.

## Key Features

- **Multi-LLM Evaluation**: Seamlessly integrate and evaluate multiple LLMs.
- **Custom LLM Integration**: Decouple model inference from evaluation, enabling you to run your preferred LLMs and evaluate the results independently.
- **Support for Approximation**: Evaluate responses from approximated or fine-tuned LLMs.
- **Modular Design**: Organize the workflow into request generation, LLM inference, and evaluation, allowing easier customization and extensibility.

## How It Works

The framework is structured around three main scripts:

1. **`generate_request.py`**: Generates evaluation requests and saves them to a file.
2. **`llama_infer.py`**: Simulates running your LLM on the generated requests.
3. **`evaluate_response.py`**: Evaluates the responses generated by your LLM against the task benchmark.

### Workflow

1. **Generate Request**:
   Use `generate_request.py` to create requests for a specified evaluation task.
2. **Run LLM**:
   Use `llama_infer.py` (or your custom script) to process the requests and generate responses.
3. **Evaluate Responses**:
   Use `evaluate_response.py` to evaluate the generated responses and get the results.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Oasis-Git/lm-evaluation-harness-adapter.git
cd lm-evaluation-harness-adapter
pip install -r requirements.txt
python setup.py install

```
## Task Verification

This section tracks the status of tasks tested with **LM Eval Adapter**. The status column uses color-coded indicators to represent the current state of each task:

- **🟢 Passed**: Task is fully functional and produces expected results.
- **🟡 Testing**: Task is under active testing.
- **🔴 Failed**: Task has known issues that need to be resolved.

| **Task Name**          | **Status**      |
|-------------------------|-----------------|
| ifeval               | 🟢 Passed       |
| gsm8k                | 🟢 Passed        |
| xsum        | 🟢 Passed       |
| triviaqa        | 🟢 Passed       |
| nq_open        | 🟢 Passed       |

Feel free to contribute to improving task support or report issues for any tasks!

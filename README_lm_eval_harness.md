# LLM Benchmark Evaluation Tool

This tool allows you to evaluate large language models (LLMs) on various benchmarks using lm-evaluation-harness.

## Features

- Supports multiple Kyrgyz benchmarks: kyrgyzMMLU, Kyrgyz_RC, HellaSwag_kg, Winogrande_kg, Truthful_kg, BoolQ_kg and GSM8K_kg
- Supports multiple English benchmarks: MMLU, HellaSwag (LSWAG), Winogrande, TruthfulQA, BoolQ and GSM8K
- Works with both Hugging Face models and local fine-tuned models
- Configurable few-shot learning and batch size
- Saves evaluation results to files


## Evaluating `meta-llama/Llama-3.2-1B-Instruct` on Kyrgyz Benchmarks using `lm-evaluation-harness`

This document describes the steps taken to set up the `lm-evaluation-harness` framework and run an evaluation of a language model on custom tasks for the Kyrgyz language.

## 1. Environment Setup and Configuration

### Steps:

#### 🔹 Cloning the Repository:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
```
#### 🔹 Installing in "Editable Mode"":
Allows you to make changes to the source code without reinstalling:
```bash
pip install -e .
```
## 2. Adding Custom Tasks for the Kyrgyz Language
By default, the framework does not include tasks in Kyrgyz. To add them:
#### 🔸 Create a Python file .py and .yaml in lm_eval/tasks
In this file, implement:
- Load the HF Dataset, for exampl,e `TTimur/kyrgyzMMLU`

- Formatting the prompt (question + options)

- Defining the correct answer

- Configuring the data splits used (test for evaluation, validation for few-shot examples)

Yaml for kyrgyz_mmlu_history:
```yaml
python_file: kyrgyz_tasks.py
task: kyrgyz_mmlu_history
class: !function kyrgyz_tasks.KyrgyzMMLUHistory
output_type: loglikelihood
repeats: 1
```

#### 🔸  Run the Evaluation
Once configured, everything is ready to run:
```bash
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
        --tasks kyrgyz_mmlu_history \
        --device cuda \
        --batch_size 4 \
        --num_fewshot 5 \
        --apply_chat_template \
        --output_path ./kyrgyz_results/results/meta-llama/harness
```
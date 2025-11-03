# LLM Benchmark Evaluation Tool

This tool allows you to evaluate large language models (LLMs) on various benchmarks using the lighteval library.

Documentation: https://huggingface.co/docs/lighteval/v0.9.2/en/quicktour

## Features

- Supports multiple Kyrgyz benchmarks: kyrgyzMMLU, Kyrgyz_RC, HellaSwag_kg, Winogrande_kg, Truthful_kg (MC), BoolQ_kg and GSM8K_kg
- Supports multiple English benchmarks: MMLU, HellaSwag (LSWAG), Winogrande, TruthfulQA, BoolQ and GSM8K
- Works with both Hugging Face models and local fine-tuned models
- Configurable few-shot learning and batch size
- Saves evaluation results to files
- Note, TruthfulQA is run as few-shot (6) in Lighteval by default. Pre-query is implemented in default_prompts.py

## Available Kyrgyz benchmarks
1. kyrgyz_mmlu_all
- kyrgyz_mmlu_literature
- kyrgyz_mmlu_history
- kyrgyz_mmlu_lang
- kyrgyz_mmlu_medicine
- kyrgyz_mmlu_biology
- kyrgyz_mmlu_geography
- kyrgyz_mmlu_math
- kyrgyz_mmlu_physics
- kyrgyz_mmlu_chemistry

2. kyrgyz_rc_all
- kyrgyz_rc_literature
- kyrgyz_rc_math
- kyrgyz_rc_news
- kyrgyz_rc_wiki

3. winogrande_kg
4. truthful_kg
5. gsm8k_kg
6. hellaswag_kg
7. boolq_kg

kyrgyz_llm_leaderboard_tasks_zeroshot.txt
```bash
# MMLU - kyrgyzMMLU
community|kyrgyz_evals:kyrgyz_mmlu_medicine|0|0
community|kyrgyz_evals:kyrgyz_mmlu_history|0|0
community|kyrgyz_evals:kyrgyz_mmlu_literature|0|0
community|kyrgyz_evals:kyrgyz_mmlu_lang|0|0
community|kyrgyz_evals:kyrgyz_mmlu_biology|0|0
community|kyrgyz_evals:kyrgyz_mmlu_chemistry|0|0
community|kyrgyz_evals:kyrgyz_mmlu_geography|0|0
community|kyrgyz_evals:kyrgyz_mmlu_math|0|0
community|kyrgyz_evals:kyrgyz_mmlu_physics|0|0
# Reading comprehension - kyrgyzRC
community|kyrgyz_evals:kyrgyz_rc_literature|0|0
community|kyrgyz_evals:kyrgyz_rc_math|0|0
community|kyrgyz_evals:kyrgyz_rc_news|0|0
community|kyrgyz_evals:kyrgyz_rc_wiki|0|0
# WinoGrande
community|kyrgyz_evals:winogrande_kg|0|0
# GSM8K
community|kyrgyz_evals:gsm8k_kg|0|0
# BoolQA
community|kyrgyz_evals:boolq_kg|0|0
# HellaSwag
community|kyrgyz_evals:hellaswag_kg|0|0
# TruthfulQA
community|kyrgyz_evals:truthfulqa_mc_kg|0|0
```

kyrgyz_llm_leaderboard_tasks_fewshot.txt

```bash
# MMLU - kyrgyzMMLU
community|kyrgyz_evals:kyrgyz_mmlu_medicine|5|0
community|kyrgyz_evals:kyrgyz_mmlu_history|5|0
community|kyrgyz_evals:kyrgyz_mmlu_literature|5|0
community|kyrgyz_evals:kyrgyz_mmlu_lang|5|0
community|kyrgyz_evals:kyrgyz_mmlu_biology|5|0
community|kyrgyz_evals:kyrgyz_mmlu_chemistry|5|0
community|kyrgyz_evals:kyrgyz_mmlu_geography|5|0
community|kyrgyz_evals:kyrgyz_mmlu_math|5|0
community|kyrgyz_evals:kyrgyz_mmlu_physics|5|0
# Reading comprehension - kyrgyzRC
community|kyrgyz_evals:kyrgyz_rc_literature|5|0
community|kyrgyz_evals:kyrgyz_rc_math|5|0
community|kyrgyz_evals:kyrgyz_rc_news|5|0
community|kyrgyz_evals:kyrgyz_rc_wiki|5|0
# WinoGrande
community|kyrgyz_evals:winogrande_kg|5|0
# GSM8K
community|kyrgyz_evals:gsm8k_kg|5|0
# BoolQA
community|kyrgyz_evals:boolq_kg|5|0
# HellaSwag
community|kyrgyz_evals:hellaswag_kg|10|0
# TruthfulQA
community|kyrgyz_evals:truthfulqa_mc_kg|5|0
```


## Lighteval Installation:

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r lighteval/requirements_lighteval.txt
export HUGGINGFACE_TOKEN=<your_hf_token>
```


## Usage of Kyrgyz benchmarks :

### Evaluating a Hugging Face model with individual Benchmark:
```python
lighteval accelerate \
"pretrained=meta-llama/Llama-3.2-1B,dtype=bfloat16,model_parallel=True" \
"community|kyrgyz_evals:kyrgyz_mmlu_medicine|5|0" \
--custom-tasks lighteval/kyrgyz_eval_llm.py \
--output-dir="lighteval/results_kyrgyz"
```

### Evaluating a local\pretrained model with individual Benchmark:
```python
lighteval accelerate \
"pretrained=/home/tim/LLM/Llama-3.1-8b-merged,dtype=bfloat16,model_parallel=True" \
"community|kyrgyz_evals:kyrgyz_mmlu_medicine|5|0" \
--custom-tasks lighteval/kyrgyz_eval_llm.py \
--output-dir="lighteval/results_kyrgyz"
```

### Evaluating a local\pretrained model on multiple GPU (ex, 8):
```python
accelerate launch --main_process_port 0 --multi_gpu --num_processes=8 \
lighteval accelerate \
"pretrained=/home/tim/LLM/Llama-3.1-8b-merged,dtype=bfloat16,model_parallel=True" \
"community|kyrgyz_evals:kyrgyz_mmlu_medicine|5|0" \
--custom-tasks lighteval/kyrgyz_eval_llm.py \
--output-dir="lighteval/results_kyrgyz"
```

### Running all benchmarks from kyrgyz_llm_leaderboard_tasks.txt
```python
lighteval accelerate \
     "model_name=meta-llama/Llama-3.2-1B,dtype=bfloat16,model_parallel=True" \
     kyrgyz_llm_leaderboard_tasks.txt \
     --custom-tasks lighteval/kyrgyz_eval_llm.py \
     --output-dir="lighteval/results_kyrgyz"
```



#### Arguments
--output-dir - saving results
kyrgyz_evals:task|num_few_shot|{0 for strict `num_few_shots`, or 1 to allow a truncation if context size is too small}


## Usage of English benchmarks

### Running all benchmarks from english_open_llm_leaderboard_tasks.txt:

```
lighteval accelerate \
     "model_name=meta-llama/Llama-3.2-1B, model_parallel=True" \
     english_open_llm_leaderboard_tasks.txt \
     --batch_size 8
     --output-dir="lighteval/results_english"
```

### Running an Individual Benchmark:
```
lighteval accelerate \
     "model_name=meta-llama/Llama-3.2-1B, model_parallel=True" \
     "leaderboard|mmlu:high_school_european_history|5|0" \
     --batch_size 8
     --output-dir="lighteval/results_english"
```

### Using english_eval_llm.py:
```python
python english_eval_llm.py --model MODEL_NAME_OR_PATH [--local_model] [--bench BENCHMARK] [--batch_size BATCH_SIZE] [--num_fewshot NUM_FEWSHOT]
```
### Arguments
- --model : Model name (Hugging Face model ID) or path to local model
- --local_model : Flag to indicate if the model is loaded from a local path
- --bench : Benchmark to run (mmlu, lswag, winogrande, gsm)
- --batch_size : Batch size for evaluation
- --num_fewshot : Number of few-shot examples


## More examples with English benchmarks:
### Evaluating a Hugging Face model
Run MMLU benchmark on Llama 3.1 8B:
```bash
python english_eval_llm.py --model meta-llama/Llama-3.1-8B --bench mmlu
```

### Evaluating a local fine-tuned model
Run GSM8K benchmark on a locally fine-tuned model:
```bash
python english_eval_llm.py --model /Users/Timur/Desktop/Projects/AKYLAI/Models/my-finetuned-llama --local_model --bench gsm
```

Run Winogrande benchmark on a local model with custom batch size:
```bash
python english_eval_llm.py --model /Users/Timur/Desktop/Projects/AKYLAI/Models/my-finetuned-llama --local_model --bench winogrande --batch_size 2
```


## Notes
- For local models, make sure they are saved in a format compatible with Hugging Face's Transformers library
- Adjust batch size based on your GPU memory capacity
- The script automatically uses GPU if available, falling back to CPU if necessary
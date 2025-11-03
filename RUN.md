# English benchmarks run

Documentation: https://huggingface.co/docs/lighteval/v0.9.2/en/quicktour

! generation_parameters={temperature=0.6, top_p=0.9} for All models

## Zero-shot running all benchmarks from english_open_llm_leaderboard_tasks_zeroshot.txt:

```
lighteval accelerate \
     "model_name=meta-llama/Llama-3.2-1B" \
     english_open_llm_leaderboard_tasks_zeroshot.txt \
     
```

## Few-shot running all benchmarks from english_open_llm_leaderboard_tasks_fewshot.txt:

```
lighteval accelerate \
     "model_name=meta-llama/Llama-3.2-1B" \
     english_open_llm_leaderboard_tasks_fewshot.txt \
     
```

## Running an Individual Benchmark:
```
lighteval accelerate \
     "model_name=meta-llama/Llama-3.2-1B" \
     "leaderboard|mmlu:high_school_european_history|5|0" \
     
```

## Running from yaml file:
```
lighteval accelerate \
     lighteval_model_template.yaml \
     english_open_llm_leaderboard_tasks_fewshot.txt \
     --output-dir="lighteval/results_english"
     
```

##  ===== All commands used for English
You can run this script once to execute all commands for all models. If you want to use a different tasks file (e.g., for zero-shot), you can adjust the TASKS variable in the script.

```
cd lighteval
chmod +x generate_model_yamls_and_bash.sh
./generate_model_yamls_and_bash.sh
```

# Kyrgyz benchmarks run

## Zero-shot running all benchmarks from kyrgyz_llm_leaderboard_tasks_zeroshot.txt
```
lighteval accelerate \
     lighteval_model_template.yaml \
     kyrgyz_llm_leaderboard_tasks_zeroshot.txt \
     --custom-tasks lighteval/kyrgyz_eval_llm.py \
     --output-dir="lighteval/results_kyrgyz"
```

## Few-shot running all benchmarks from kyrgyz_llm_leaderboard_tasks_fewshot.txt
```
lighteval accelerate \
     lighteval_model_template.yaml \
     kyrgyz_llm_leaderboard_tasks_fewshot.txt \
     --custom-tasks lighteval/kyrgyz_eval_llm.py \
     --output-dir="lighteval/results_kyrgyz"
```

## Running an Individual Benchmark:
```
!lighteval accelerate \
lighteval_model_template.yaml \
"community|kyrgyz_evals:kyrgyz_mmlu_medicine|5|0" \
--custom-tasks lighteval/kyrgyz_eval_llm.py \
--output-dir="lighteval/results_kyrgyz"
```


##  ===== All commands used for Kyrgyz
You can run this script once to execute all commands for all models. If you want to use a different tasks file (e.g., for zero-shot), you can adjust the TASKS variable in the script.

```
cd lighteval
!chmod +x generate_model_yamls_and_bash_kyrgyz.sh
./generate_model_yamls_and_bash_kyrgyz.sh
```
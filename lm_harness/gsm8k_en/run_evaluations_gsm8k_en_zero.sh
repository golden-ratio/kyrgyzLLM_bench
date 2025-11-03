#!/bin/bash

# Define the list of models to evaluate
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "google/gemma-3-4b-it"
    "google/gemma-3-1b-it"
    "google/gemma-3-270m-it"
)

# Define common evaluation parameters
TASK="gsm8k"
OUTPUT_DIR="/home/jovyan/kyrgyz_llm_eval-main/lm_harness_results/en/zero"
BATCH_SIZE=8
GEN_KWARGS="do_sample=True,temperature=0.6,top_p=0.9"
NUM_FEWSHOT=0
LOG_FILE="${OUTPUT_DIR}/evaluation_log.txt"

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# --- Start Loop ---
echo "Starting evaluation loop for ${#MODELS[@]} models..." | tee -a "${LOG_FILE}"

# Loop through each model in the array
for MODEL_NAME in "${MODELS[@]}"; do
    
    # Sanitize model name for use in file names (replaces slashes with underscores)
    SAFE_MODEL_NAME=$(echo "${MODEL_NAME}" | tr / _)

    echo "" | tee -a "${LOG_FILE}"
    echo "--- Starting evaluation for: ${MODEL_NAME} ---" | tee -a "${LOG_FILE}"
    
    CURRENT_OUTPUT_PATH="${OUTPUT_DIR}/${SAFE_MODEL_NAME}_${TASK}"
    
    # Construct the base lm_eval command
    # *** FIX IS HERE: Removed single quotes around ${MODEL_NAME} ***
    COMMAND="accelerate launch -m lm_eval \
        --model hf \
        --tasks ${TASK} \
        --model_args pretrained=${MODEL_NAME},parallelize=True \
        --apply_chat_template \
        --batch_size ${BATCH_SIZE} \
        --gen_kwargs ${GEN_KWARGS} \
        --num_fewshot ${NUM_FEWSHOT} \
        --output_path ${CURRENT_OUTPUT_PATH}"

    # Execute the command and log the output/errors
    echo "Running command: ${COMMAND}" | tee -a "${LOG_FILE}"
    
    ${COMMAND} 2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        echo "✅ Evaluation for ${MODEL_NAME} completed successfully." | tee -a "${LOG_FILE}"
    else
        echo "❌ Evaluation for ${MODEL_NAME} FAILED. Check log for details." | tee -a "${LOG_FILE}"
    fi
    
done

echo "" | tee -a "${LOG_FILE}"
echo "All evaluations finished. Check ${LOG_FILE} for summary." | tee -a "${LOG_FILE}"
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
TASK="kyrgyz_gsm8k"
TASK_INCLUDE_PATH="."  # Path to include custom tasks (like kyrgyz_gsm8k)
DEVICE="cuda"
BATCH_SIZE=4
NUM_FEWSHOT=0
OUTPUT_DIR="/home/jovyan/kyrgyz_llm_eval-main/lm_harness_results/kg/zero"
LOG_FILE="${OUTPUT_DIR}/zero_shot_evaluation_log.txt"

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# --- Start Loop ---
echo "Starting ZERO-SHOT evaluation loop for ${#MODELS[@]} models on ${TASK}..." | tee -a "${LOG_FILE}"

# Loop through each model in the array
for MODEL_NAME in "${MODELS[@]}"; do
    
    # Sanitize model name for use in file names (replaces slashes with underscores)
    SAFE_MODEL_NAME=$(echo "${MODEL_NAME}" | tr / _)

    echo "" | tee -a "${LOG_FILE}"
    echo "--- Starting evaluation for: ${MODEL_NAME} (Zero-Shot) ---" | tee -a "${LOG_FILE}"
    
    # Define the output file path for this specific model and task
    # lm-eval will append the .json extension and other details automatically
    CURRENT_OUTPUT_PATH="${OUTPUT_DIR}/${SAFE_MODEL_NAME}_${TASK}_nf${NUM_FEWSHOT}"
    
    # Construct the lm_eval command
    COMMAND="accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL_NAME} \
        --include_path ${TASK_INCLUDE_PATH} \
        --tasks ${TASK} \
        --device ${DEVICE} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --apply_chat_template \
        --output_path ${CURRENT_OUTPUT_PATH}"

    # Execute the command and log the output/errors
    echo "Running command: ${COMMAND}" | tee -a "${LOG_FILE}"
    
    # Execute the evaluation command
    ${COMMAND} 2>&1 | tee -a "${LOG_FILE}"
    
    # Check the exit status of the last command ($?)
    if [ $? -eq 0 ]; then
        echo "✅ Evaluation for ${MODEL_NAME} completed successfully." | tee -a "${LOG_FILE}"
    else
        echo "❌ Evaluation for ${MODEL_NAME} FAILED. Check log for details." | tee -a "${LOG_FILE}"
    fi
    
done

echo "" | tee -a "${LOG_FILE}"
echo "All ZERO-SHOT evaluations finished. Check ${LOG_FILE} for summary." | tee -a "${LOG_FILE}"
#!/bin/bash

# This script generates YAML files for each model and runs lighteval on each benchmark task.
# Results and logs are saved in structured directories.
# Only missing benchmark results are executed.

MODELS_FILE="../llms_list_open.yaml"
TASKS_FILE="../kyrgyz_llm_leaderboard_tasks_zeroshot.txt"
TEMPLATE="lighteval_model_template.yaml"
YAML_DIR="kyrgyz_yaml"
RESULTS_BASE_DIR="results_kyrgyz/zeroshot"
LOG_FILE="lighteval_run.log"
CUSTOM_TASKS_ARG="lighteval/kyrgyz_eval_llm.py"

mkdir -p "$YAML_DIR"
mkdir -p "$RESULTS_BASE_DIR"

# Clear previous log
echo "Lighteval run started at $(date)" > "$LOG_FILE"

# Read all benchmarks into an array
mapfile -t TASKS < "$TASKS_FILE"

# Process each model
while IFS= read -r MODEL_NAME; do
  [[ -z "$MODEL_NAME" ]] && continue  # Skip empty lines

  SAFE_NAME=$(echo "$MODEL_NAME" | tr '/:' '__')
  YAML_FILE="$YAML_DIR/model_${SAFE_NAME}.yaml"
  MODEL_RESULTS_DIR="$RESULTS_BASE_DIR/$SAFE_NAME"

  mkdir -p "$MODEL_RESULTS_DIR"

  # Create or overwrite YAML config for model
  awk -v model="$MODEL_NAME" '{gsub(/model_name: ".*"/, "model_name: \"" model "\""); print}' "$TEMPLATE" > "$YAML_FILE"
  echo "Prepared YAML for model: $MODEL_NAME" >> "$LOG_FILE"

  ALL_COMPLETE=true

  for TASK in "${TASKS[@]}"; do
    [[ "$TASK" =~ ^# ]] && continue  # Skip comment lines
    TASK_NAME=$(basename "$TASK")
    OUTPUT_FILE="$MODEL_RESULTS_DIR/${TASK_NAME}.json"

    if [[ -f "$OUTPUT_FILE" ]]; then
      echo "[$MODEL_NAME][$TASK_NAME] Result exists. Skipping." >> "$LOG_FILE"
      continue
    fi

    ALL_COMPLETE=false
    echo "[$MODEL_NAME][$TASK_NAME] Running benchmark..." >> "$LOG_FILE"

    
    # Run lighteval for a single benchmark
    if lighteval accelerate "$YAML_FILE" "$TASK" --custom-tasks="$CUSTOM_TASKS_ARG"  --output-dir="$MODEL_RESULTS_DIR" >> "$LOG_FILE" 2>&1; then
      echo "[$MODEL_NAME][$TASK_NAME] ✅ Success" >> "$LOG_FILE"
    else
      echo "[$MODEL_NAME][$TASK_NAME] ❌ Failed" >> "$LOG_FILE"
    fi
  done

  if [[ "$ALL_COMPLETE" == true ]]; then
    echo "[$MODEL_NAME] All benchmarks already completed." >> "$LOG_FILE"
  fi

  # Clean up Hugging Face cache for this model
  MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub/models--$(echo "$MODEL_NAME" | sed 's|/|--|g')"
  if [ -d "$MODEL_CACHE_DIR" ]; then
    rm -rf "$MODEL_CACHE_DIR"
    echo "[$MODEL_NAME] Cache cleared at $MODEL_CACHE_DIR" >> "$LOG_FILE"
  fi

done < "$MODELS_FILE"

echo "Lighteval run completed at $(date)" >> "$LOG_FILE"
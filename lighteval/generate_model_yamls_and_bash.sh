#!/bin/bash

# This script will generate a YAML for each model and run lighteval for each

MODELS_FILE="../llms_list_open.yaml"
TEMPLATE="lighteval_model_template.yaml"
TASKS="../english_open_llm_leaderboard_tasks_fewshot.txt" # change to zeroshot or fewshot
RESULTS_DIR="results_english"
YAML_DIR="english_yaml"

mkdir -p "$RESULTS_DIR"
mkdir -p "$YAML_DIR"

while IFS= read -r MODEL_NAME; do
  # Skip empty lines
  [[ -z "$MODEL_NAME" ]] && continue
  # Generate a safe filename
  SAFE_NAME=$(echo "$MODEL_NAME" | tr '/:' '__')
  YAML_FILE="$YAML_DIR/model_${SAFE_NAME}.yaml"
  # Create YAML for this model
  awk -v model="$MODEL_NAME" '{gsub(/model_name: ".*"/, "model_name: \"" model "\""); print}' "$TEMPLATE" > "$YAML_FILE"
  # Run lighteval
  lighteval accelerate \
    "$YAML_FILE" \
    "$TASKS" \
    --output-dir="$RESULTS_DIR/$SAFE_NAME"
done < "$MODELS_FILE" 
#!/bin/bash

# ================= Parameters =================
BASE_DIR="./data"
GPU_ID="0"
EMBEDDING_PREFIX="bge"
TARGET_MODEL="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
# ==============================================

source $(dirname "$0")/topic.sh

echo "======================================================"
echo " STEP 1: Preprocessing & Clean RAG Evaluation"
echo "======================================================"

execute_prep() {
    local category=$1
    local topic=$2

    echo "[Prep] Processing Category: $category | Topic: $topic"
    
    python src/prep/create_contextualized_nodes.py \
        --category "$category" \
        --topic "$topic" \
        --data_dir "$BASE_DIR" \
        --model_path "BAAI/bge-large-en-v1.5"

    python src/prep/evaluate_clean_rag.py \
        --category "$category" \
        --topic "$topic" \
        --base_dir "$BASE_DIR" \
        --target_model "$TARGET_MODEL" \
        --embed_prefix "$EMBEDDING_PREFIX" \
        --gpu_id "$GPU_ID"
}

# Iterate over categories and their specific topics
for topic in "${TOPICS_POLITICS[@]}"; do execute_prep "politics" "$topic"; done
for topic in "${TOPICS_SPORTS[@]}"; do execute_prep "sports" "$topic"; done
for topic in "${TOPICS_ENTERTAINMENT[@]}"; do execute_prep "entertainment" "$topic"; done
for topic in "${TOPICS_SOCIETY_BUSINESS[@]}"; do execute_prep "society_business" "$topic"; done

echo "Step 1 Completed."
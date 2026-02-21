#!/bin/bash

# ================= Parameters =================
BASE_DIR="./data"
STANCE="oppose"
BUDGET=10
EMBEDDING_PREFIX="bge"
# ==============================================

source $(dirname "$0")/topic.sh

echo "======================================================"
echo " STEP 2: Graph Build & 2-Stage Clustering"
echo "======================================================"

execute_graph() {
    local category=$1
    local topic=$2

    echo "[Graph] Processing Category: $category | Topic: $topic"

    python src/graph_cluster/build_polarity_graph.py \
        --category "$category" \
        --topic "$topic" \
        --embedding_prefix "$EMBEDDING_PREFIX" \
        --base_dir "$BASE_DIR" \
        --stance "$STANCE"

    python src/graph_cluster/two_stage_clustering.py \
        --category "$category" \
        --topic "$topic" \
        --stance "$STANCE" \
        --base_dir "$BASE_DIR" \
        --budget "$BUDGET" \
        --model_path "BAAI/bge-large-en-v1.5"
}

for topic in "${TOPICS_POLITICS[@]}"; do execute_graph "politics" "$topic"; done
for topic in "${TOPICS_SPORTS[@]}"; do execute_graph "sports" "$topic"; done
for topic in "${TOPICS_ENTERTAINMENT[@]}"; do execute_graph "entertainment" "$topic"; done
for topic in "${TOPICS_SOCIETY_BUSINESS[@]}"; do execute_graph "society_business" "$topic"; done

echo "Step 2 Completed."
#!/bin/bash

# ================= Parameters =================
BASE_DIR="./data"
STANCE="oppose"
BUDGET=10
# ==============================================

source $(dirname "$0")/topic.sh

echo "======================================================"
echo " STEP 3: DiscourseFlip Optimization"
echo "======================================================"

execute_attack() {
    local category=$1
    local topic=$2

    echo "[Attack] Processing Category: $category | Topic: $topic"

    python src/attack/generation_manipulation_optimizer.py \
        --category "$category" \
        --topic "$topic" \
        --stance "$STANCE" \
        --base_dir "$BASE_DIR" \
        --max_poisoned_docs "$BUDGET"

    python src/attack/seo_optimizer.py \
        --category "$category" \
        --topic "$topic" \
        --stance "$STANCE" \
        --base_dir "$BASE_DIR" \
        --budget "$BUDGET"
}

for topic in "${TOPICS_POLITICS[@]}"; do execute_attack "politics" "$topic"; done
for topic in "${TOPICS_SPORTS[@]}"; do execute_attack "sports" "$topic"; done
for topic in "${TOPICS_ENTERTAINMENT[@]}"; do execute_attack "entertainment" "$topic"; done
for topic in "${TOPICS_SOCIETY_BUSINESS[@]}"; do execute_attack "society_business" "$topic"; done

echo "Step 3 Completed."
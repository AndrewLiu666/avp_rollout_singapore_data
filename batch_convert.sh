#!/bin/bash

INPUT_ROOT="./data"
OUTPUT_DIR="./data/processed_episodes"
EPISODE_ID=0

mkdir -p "$OUTPUT_DIR"

for EPISODE_PATH in "$INPUT_ROOT"/episode_*; do
    EPISODE_NAME=$(basename "$EPISODE_PATH")
    JSON_FILE="$EPISODE_PATH/data.json"

    if [ ! -f "$JSON_FILE" ]; then
        echo "Skipping $EPISODE_NAME: no data.json found"
        continue
    fi

    echo "Processing $EPISODE_NAME (ID: $EPISODE_ID)..."

    python3 ./act/convert_json_to_hdf5.py \
        --json_file "$JSON_FILE" \
        --image_dir "$EPISODE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --episode_id "$EPISODE_ID" \
        --num_episodes 1 \
        --cameras agentview 

    EPISODE_ID=$((EPISODE_ID + 1))
done

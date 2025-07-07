#!/usr/bin/env bash
# run_case1.sh
# Full shell script to train and predict on gan_case1.csv using case1 (splits 3 & 4) with pre.py

# --- Configuration ---
# Path to the CSV (no header)
CSV_FILE="./gan_case1.csv"
# Validation case: splits 3 & 4
CASE="case1"

# --- Hyperparameters (defaults, override by exporting env vars) ---
NROWS=${NROWS:-240}
SPLIT_SIZE=${SPLIT_SIZE:-10}
NUM_SPLITS=4
WINDOW_SIZE=${WINDOW_SIZE:-2}
STEP_SIZE=${STEP_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-5}
EPOCHS=${EPOCHS:-100}
LR=${LR:-0.005}
HIDDEN_SIZE=${HIDDEN_SIZE:-64}
DROPOUT=${DROPOUT:-0.1}
N_SAMPLES=${N_SAMPLES:-100}
SEED=${SEED:-42}

# --- Check dependencies ---
if [[ ! -f prediction.py ]]; then
    echo "Error: prediction.py not found in current directory."
    exit 1
fi

# --- Execute Python script ---
python3 prediction.py \
    --csv_file "$CSV_FILE" \
    --nrows "$NROWS" \
    --split_size "$SPLIT_SIZE" \
    --num_splits $NUM_SPLITS \
    --window_size "$WINDOW_SIZE" \
    --step_size "$STEP_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --hidden_size "$HIDDEN_SIZE" \
    --dropout "$DROPOUT" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED" \
    --case "$CASE"

echo "Prediction for $CASE complete on $CSV_FILE."

#!/bin/bash

NUM_ENSEMBLES=100
MODEL_NAME="minixception"
OPTIMIZER="adamw"

echo "Starting training for $NUM_ENSEMBLES ensembles"

for i in $(seq 0 $((NUM_ENSEMBLES - 1)))
do
  seed_value=$i
  label_string="ensemble_$i"

  echo "----------------------------------------"
  echo "Running Ensemble $i (Seed: $seed_value, Label: $label_string):"
  echo "----------------------------------------"
  python3 train.py --seed "$seed_value" --label "$label_string" --model "$MODEL_NAME" --optimizer "$OPTIMIZER"
  sleep 1
done

echo "----------------------------------------"
echo "All $NUM_ENSEMBLES ensemble trainings complete."

#!/bin/bash

NUM_ENSEMBLES=20

echo "Starting training for $NUM_ENSEMBLES ensembles"

for i in $(seq 0 $((NUM_ENSEMBLES - 1)))
do
  seed_value=$i
  label_string="ensemble_$i"

  echo "----------------------------------------"
  echo "Running Ensemble $i (Seed: $seed_value, Label: $label_string):"
  echo "Command: python3 train.py --seed $seed_value --label $label_string"
  echo "----------------------------------------"
  python3 train.py --seed "$seed_value" --label "$label_string"
  sleep 1
done

echo "----------------------------------------"
echo "All $NUM_ENSEMBLES ensemble trainings complete."

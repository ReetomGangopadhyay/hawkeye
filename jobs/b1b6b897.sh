#!/bin/bash

#$ -N exp_b1b6b897
#$ -pe omp 4
#$ -l h_rt=00:15:00
#$ -l gpus=1
#$ -j y

echo "Experiment: b1b6b897"
echo "Host: $(hostname)"
echo "Start: $(date)"

python worker.py     --cpus 4     --gpus 1     --batch-size 128     --workers 4     --output results/b1b6b897.csv     --command "python train.py --batch-size 128 --workers 4"

echo "End: $(date)"

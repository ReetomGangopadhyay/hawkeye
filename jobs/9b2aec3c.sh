#!/bin/bash

#$ -N exp_9b2aec3c
#$ -pe omp 4
#$ -l h_rt=00:15:00
#$ -l gpus=1
#$ -j y

echo "Experiment: 9b2aec3c"
echo "Host: $(hostname)"
echo "Start: $(date)"

python worker.py     --cpus 4     --gpus 1     --batch-size 128     --workers 4     --output results/9b2aec3c.csv     --command "python train.py --batch-size 128 --workers 4"

echo "End: $(date)"

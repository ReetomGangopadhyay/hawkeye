#!/bin/bash

#$ -N exp_c18eef64
#$ -pe omp 4
#$ -l h_rt=00:15:00
#$ -l gpus=1
#$ -j y

echo "Experiment: c18eef64"
echo "Host: $(hostname)"
echo "Start: $(date)"

python worker.py     --cpus 4     --gpus 1     --batch-size 128     --workers 4     --output results/c18eef64.csv     --command "python train.py --batch-size 128 --workers 4"

echo "End: $(date)"

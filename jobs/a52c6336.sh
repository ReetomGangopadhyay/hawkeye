#!/bin/bash

#$ -N exp_a52c6336
#$ -pe omp 4
#$ -l h_rt=00:15:00
#$ -l gpus=1
#$ -j y

echo "Experiment: a52c6336"
echo "Host: $(hostname)"
echo "Start: $(date)"

python worker.py     --cpus 4     --gpus 1     --batch-size 128     --workers 4     --output results/a52c6336.csv     --command "python train.py --batch-size 128 --workers 4 --steps 200"

echo "End: $(date)"

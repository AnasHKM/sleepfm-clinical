#!/bin/bash
#SBATCH --job-name=train-model
#SBATCH --partition=bch-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anas.hakim@childrens.harvard.edu
#SBATCH --mem-per-gpu=46G
#SBATCH --time=7-00:00:00
#SBATCH --output=./logs/training%j.out

source ~/.bashrc
conda activate sleepfm_env

python trainer.py
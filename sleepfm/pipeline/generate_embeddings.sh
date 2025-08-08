#!/bin/bash
#SBATCH --job-name=Generate-embedding
#SBATCH --partition=bch-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anas.hakim@childrens.harvard.edu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=46G
#SBATCH --time=7-00:00:00
#SBATCH --output=../logs/generate_embedding%j.out

source ~/.bashrc
conda activate sleepfm_env

python generate_embeddings.py
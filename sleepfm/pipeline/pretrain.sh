#!/bin/bash
#SBATCH --job-name=Transformer-pretrain
#SBATCH --partition=bch-gpu
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anas.hakim@childrens.harvard.edu
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=46G
#SBATCH --time=7-00:00:00
#SBATCH --output=../logs/pretraining_%j.out

source ~/.bashrc
conda activate sleepfm_env

fusermount -uz /home/ch266186/data/s3_processed

s3fs ga-sandbox-local-s3:/preprocessed ~/data/s3_processed   -o profile=default   -o use_path_request_style   -o url=https://s3.amazonaws.com   -o compat_dir

python pretrain.py
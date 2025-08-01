#!/bin/bash
#SBATCH --job-name=edf_to_hdf5
#SBATCH --partition=bch-gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anas.hakim@childrens.harvard.edu
#SBATCH --output=../logs/edf_to_hdf5_%j.out

source ~/.bashrc
conda activate sleepfm_env


python preprocessing_aws.py --num_threads 10 --resample_rate 128
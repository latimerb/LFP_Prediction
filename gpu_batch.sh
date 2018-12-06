#!/bin/bash

#SBATCH --job-name=gpu_LSTM
#SBATCH --output=LSTM%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time 0-01:00

#SBATCH --partition Gpu
#SBATCH --gres gpu:1

source ~/data/virtualenvs/keras/keras_virtual_env/bin/activate

echo "Starting at: $(date)"

module load cuda/cuda-9.0.176
module load cudnn/cudnn-7.1.4-cuda-9.0.176

module list

python LSTM_ED_Multivar.py

echo "Done at: $(date)"

deactivate

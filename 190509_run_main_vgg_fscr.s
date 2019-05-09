#!/bin/bash

#SBATCH --job-name=vgg_raw
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('vgg_raw)"
python 190509_main_vgg_fscr.py --model vgg --batch-size 64 --save 190509_vgg_raw.pt --epochs 100 --feature-pinning False
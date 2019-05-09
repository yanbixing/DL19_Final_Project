#!/bin/bash

#SBATCH --job-name=vgg_AEB_FH2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('vgg_AE_based_FH2')"
python 190509_main_vgg_AEbased_FreeH2.py --model vgg --AE-file 190507SDvggAE_D01_lr005.pt --batch-size 256 --feature-pinning True --save 190509_vgg_AEB_lr005_FH2.pt --epochs 500 --lr 0.001
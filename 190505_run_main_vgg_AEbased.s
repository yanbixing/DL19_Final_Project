#!/bin/bash

#SBATCH --job-name=vgg_AEB
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('vgg_AE_based')"
python 190505_main_vgg_AEbased.py --model vgg --AE-file 190505SDvggAE_D01_lr001.pt --batch-size 256 --feature-pinning True --save 190507_vgg_AEB_lr001.pt --epochs 500 --lr 0.01
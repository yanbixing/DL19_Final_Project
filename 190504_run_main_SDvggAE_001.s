#!/bin/bash

#SBATCH --job-name=SDvggAE
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_SDvggAE_model')"
python 1.90504_main_SDvggAE.py --model vgg --batch-size 512 --save 190504_SDvggAE_D01_lr001.pt --epochs 40 --lr 0.01
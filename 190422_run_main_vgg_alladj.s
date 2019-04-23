#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32GB
#SBATCH --time=72:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_vgg_alladj_model')"
python 190422_main_vgg_alladj.py --model vgg --save 190422_vgg_alladj.pt --epochs 100 --pretrained True
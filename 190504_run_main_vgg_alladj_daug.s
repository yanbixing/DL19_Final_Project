#!/bin/bash

#SBATCH --job-name=vgg_daug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_vgg_alladj_daug_model')"
python 190504_main_vgg_alladj_daug.py --model vgg --save 190504_vgg_alladj_daug.pt --epochs 50 --pretrained True
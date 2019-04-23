#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=60:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_vgg_fcadj_model')"
python 190422_main_vgg_fcadj.py --model vgg --save 190422_vgg_fcadj.pt --epochs 60 --pretrained True
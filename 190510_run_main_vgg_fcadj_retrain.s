#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=60:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_vgg_fcadj_model')"
python 190422_main_vgg_fcadj.py --model vgg --model-file 190422_vgg_fcadj.pt --save 190510_vgg_fcadj_retrain.pt --epochs 60 --pretrained False
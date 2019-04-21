#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_raw_vgg_model')"
python 190421_main_raw_md_pfm.py --model vgg --save 190421_raw_vgg.pt --epochs 20 --pretrained True
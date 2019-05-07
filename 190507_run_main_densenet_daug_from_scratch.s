#!/bin/bash

#SBATCH --job-name=DN121_daug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_densenet_daug')"
python 190507_main_densenet_daug.py --model densenet  --model-file 190506_main_densenet_fscr_hflip.pt --batch-size 64 --save 190507_main_densenet_daug.pt --epochs 100 --feature-pinning False
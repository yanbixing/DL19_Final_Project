#!/bin/bash

#SBATCH --job-name=DN161_fscr_hflip
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('DN161_fscr_hflip')"
python 190506_main_densenet161_fscr_hflip.py --model densenet --model-folder --batch-size 128 --save 190506_main_densenet161_fscr_hflip.pt --epochs 100 --feature-pinning False
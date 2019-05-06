#!/bin/bash

#SBATCH --job-name=DN121_fscr_hflip
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('DN121_fscr_hflip')"
python 190506_main_densenet_fscr_hflip.py --model densenet --batch-size 128 --save 190506_main_densenet_fscr_hflip.pt --epochs 100 --feature-pinning False
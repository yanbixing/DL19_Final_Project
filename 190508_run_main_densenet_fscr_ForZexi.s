#!/bin/bash

#SBATCH --job-name=DN121_fscr_hflip
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

module purge
source XXXXXXXXXXXXXXXXXXX

python -c "print('DN121_fscr_hflip')"
python 190508_main_densenet_fscr_ForZexi.py --model densenet --dataset-path XXXXXXXXX  --save-folder XXXXXXXX --save XXXXXXXXXX.pt --batch-size 64 --epochs 100 --feature-pinning False
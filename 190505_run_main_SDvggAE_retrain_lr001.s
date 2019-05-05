#!/bin/bash

#SBATCH --job-name=SDvggAE_001
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=24GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('190505_main_SDvggAE_retrain.py')"
python 190505_main_SDvggAE_retrain.py --model vgg --model-folder /scratch/by783/DL_Final_models/ --model-file 190504_SDvggAE_D01_lr001.pt --batch-size 512 --save 190505SDvggAE_D01_lr001.pt --epochs 150 --lr 0.01
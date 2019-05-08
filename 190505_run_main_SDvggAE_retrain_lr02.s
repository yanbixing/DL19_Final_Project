#!/bin/bash

#SBATCH --job-name=SDvggAE_02
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('190505_main_SDvggAE_retrain.py')"
python 190505_main_SDvggAE_retrain.py --model vgg --model-folder /beegfs/by783/DL_Final_models/ --model-file 190507SDvggAE_D01_lr02 --batch-size 1024 --save 190508SDvggAE_D01_lr02.pt --epochs 100 --lr 0.2 --momentum 0.5
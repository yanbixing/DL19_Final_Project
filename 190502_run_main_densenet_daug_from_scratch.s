#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=96:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_densenet_fromscratch')"
python 190502_main_densenet_daug.py --model densenet  --model-file 190421_raw_densenet.pt --batch-size 128 --save-folder /scratch/by783/DL_Final_models/ --save 190502_main_densenet_daug_fromscratch.pt --epochs 100 --feature-pinning False
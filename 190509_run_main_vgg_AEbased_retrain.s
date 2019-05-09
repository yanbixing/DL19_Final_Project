#!/bin/bash

#SBATCH --job-name=vgg_AEB_re
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('vgg_AE_based_re')"
python 190509_main_vgg_AEbased_retrain.py --model vgg --model-file 190508_vgg_AEB_lr005.pt --batch-size 256 --feature-pinning False --save 190509_vgg_AEB_lr005_re.pt --epochs 500 --lr 0.001
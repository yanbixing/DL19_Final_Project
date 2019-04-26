#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task=10
#SBATCH --mem=36GB
#SBATCH --time=72:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_raw_vggae_en_pretrained_model')"
python 190425_main_raw_vggae.py --model vgg --batch-size 768 --save 190425_raw_vggae_en_pretrained.pt --epochs 50 --pretrained True
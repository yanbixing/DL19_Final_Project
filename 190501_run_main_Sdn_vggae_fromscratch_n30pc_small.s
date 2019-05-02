#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_SDn_vggae_fromscratch_30pc_model')"
python 190501_main_SDn_vggae.py --model vgg --batch-size 256 --save 190501_SDn_vggae_fromscratch_30pc.pt --epochs 40 --pretrained False --noise-level 0.3
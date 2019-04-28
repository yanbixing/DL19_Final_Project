#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_dn_vggae_fromscratch_model')"
python 190427_main_dn_vggae.py --model vgg --batch-size 512 --save 190425_dn_vggae_fromscratch_30pc.pt --epochs 40 --pretrained False --noise-level 0.066
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=96:00:00

module purge
source ~/myenv/bin/activate

python -c "print('begin_raw_vggae_fromscratch_model')"
python 190425_main_raw_vggae.py --model vgg --batch-size 512 --save 190425_raw_vggae_fromscratch_s.pt --epochs 40 --pretrained False
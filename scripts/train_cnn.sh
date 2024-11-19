#!/bin/bash
#SBATCH -t 0-03:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:1
#SBATCH --mem-per-cpu=8000

# initialize the environment
# (adapt to your setup)
module load rocky8-spack
module load python/3.9.7-gcc-10.3.0-kxnt
. venv/torch3/bin/activate

echo "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=${HOME}/chr2024

cd chr2024/cnn
for model in resnet18 resnet50 densenet121 mobilenet_v2; do
  for i in 0 1 2 3 4; do
    mkdir -p ${model}_reports/${i}
  done
done
#mkdir -p states

# uncomment according to your compute capacity
#for i in 0 1 2 3 4; do
for i in 0; do
  python train.py resnet18 resnet18_reports/${i} ../images/split_data/int_ext${i}/
  #python train.py resnet50 resnet50_reports/${i} ../images/split_data/int_ext${i}/
  #python train.py densenet121 densenet121_reports/${i} ../images/split_data/int_ext${i}/
  #python train.py mobilenet_v2 mobilenet_v2_reports/${i} ../images/split_data/int_ext${i}/
done

#for i in 0 1 2 3 4; do
  #python train.py resnet18 resnet18_reports/${i} ../images/split_data/gro_rai_aer${i}/
  #python train.py resnet50 resnet50_reports/${i} ../images/split_data/gro_rai_aer${i}/
  #python train.py densenet121 densenet121_reports/${i} ../images/split_data/gro_rai_aer${i}/
  #python train.py mobilenet_v2 mobilenet_v2_reports/${i} ../images/split_data/gro_rai_aer${i}/
#done

#!/bin/bash
#SBATCH -t 0-01:30
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:1
#SBATCH --mem-per-cpu=16000

# initialize the environment
# (adapt to your setup)
module load rocky8-spack
module load python/3.9.7-gcc-10.3.0-kxnt
. venv/torch3/bin/activate

echo "$CUDA_VISIBLE_DEVICES"
cd chr2024/multimodal
export PYTHONPATH=${HOME}/chr2024

for exp_name in int_ext gro_rai_aer; do
  for i in 0 1 2 3 4; do
    mkdir -p reports/${exp_name}${i}
  done
done

python classify.py openai/clip-vit-base-patch16 ../images/split_data/ prompts_db.json reports/
#python classify.py openai/clip-vit-base-patch32 ../images/split_data/ prompts_db.json reports/
#python classify.py openai/clip-vit-large-patch14 ../images/split_data/ prompts_db.json reports/
#python classify.py google/siglip-base-patch16-224 ../images/split_data/ single_prompt.json reports/
#python classify.py google/siglip-base-patch16-224 ../images/split_data/ prompts_db.json reports/
#python classify.py google/siglip-large-patch16-256 ../images/split_data/ prompts_db.json reports/
#python classify.py google/siglip-so400m-patch14-384 ../images/split_data/ prompts_db.json reports/


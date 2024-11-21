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
# change to this for BLIP-2, if LAVIS was installed separately
#. venv/lavis2/bin/activate

echo "$CUDA_VISIBLE_DEVICES"
cd chr2024/search
export PYTHONPATH=${HOME}/chr2024

mkdir -p reports

# uncomment as needed
#for model in openai/clip-vit-base-patch32 openai/clip-vit-base-patch16 \
# openai/clip-vit-large-patch14 google/siglip-base-patch16-224 \
# google/siglip-large-patch16-256 google/siglip-so400m-patch14-384; do
for model in google/siglip-so400m-patch14-384; do
  python gen_search_results.py ${model} ../images/search/ ../images/json/search_terms.json reports
done

#python gen_search_results.py blip2 ../images/search/ ../images/json/search_terms.json reports

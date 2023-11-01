#!/bin/bash
#SBATCH --account=laion
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH --job-name=laion
#SBATCH --output=llava_wds-multi.out
#SBATCH --error=llava_wds-multi.err
#SBATCH --mail-type=ALL

source ~/llava-venv-39/bin/activate

cd /fsx/home-peterbevan/LLaVA

git checkout main

python3 run_llava_wds-multi-jdb.py --start_tar 100 --end_tar 119 --dataset "JourneyDB"

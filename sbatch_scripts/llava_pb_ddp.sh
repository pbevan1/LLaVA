#!/bin/bash
#SBATCH --account=laion
#SBATCH --partition=g40x
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH --job-name=laion
#SBATCH --output=llava_wds-multi.out
#SBATCH --error=llava_wds-multi.err
#SBATCH --mail-type=ALL

export MASTER_PORT=12340
export WORLD_SIZE=16

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/llava-venv/bin/activate

cd /fsx/home-peterbevan/LLaVA

git checkout main

python3 run_llava_wds-multi-jdb.py

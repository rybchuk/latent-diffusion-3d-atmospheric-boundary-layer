#!/bin/bash
#SBATCH --nodes=1          #
#SBATCH --time=24:00:00    # Set time limit
#SBATCH --account=XX
#SBATCH --ntasks=1         # Maximum CPU cores for job
#SBATCH --gres=gpu:2	   # GPU request
#SBATCH --output=slurm/logs/job_output_filename.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XX

module load conda
conda activate XX

CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/autoencoder/ae_32_32_16.yaml -t --gpus 0,1
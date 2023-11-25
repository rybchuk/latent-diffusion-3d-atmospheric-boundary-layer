#!/bin/bash
#SBATCH --nodes=1          #
#SBATCH --time=24:00:00    # Set time limit
# #SBATCH --time=04:00:00    # Set time limit
# #SBATCH --partition=short
# #SBATCH --time=00:59:00
# #SBATCH --partition=debug
#SBATCH --account=XX
#SBATCH --ntasks=1         # Maximum CPU cores for job
#SBATCH --gres=gpu:2	   # GPU request
#SBATCH --output=slurm/logs/job_output_filename.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XX

module load conda
conda activate XX

CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/inpaint-box.yaml -t --gpus 0,1 --train False --no-test False --gpus 0,1 --name inpaint_box --resume_from_checkpoint models/ldm/epoch=000023.ckpt

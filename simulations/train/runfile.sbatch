#!/bin/bash
#SBATCH --nodes=4
#SBATCH --account=wakedynamics
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alex.rybchuk@gmail.com
#SBATCH --output=logs/job_output_filename.%j.out  # %j will be replaced with the job ID
# #SBATCH --partition=debug
# #SBATCH --time=00:59:00
# #SBATCH --partition=short
# #SBATCH --time=03:59:00
#SBATCH --partition=standard
#SBATCH --time=47:59:00

module purge
#module load gcc
#module load mpt
#module load cmake
#module load netcdf-c/4.7.3
module load gcc/8.4.0 mpt mkl cmake

export EXAWIND_DIR=/nopt/nrel/ecom/exawind/exawind-2020-09-21/install/gcc
export MPI_TYPE_DEPTH=15
export MPI_IB_CONGESTED=true
export MPI_XPMEM_ENABLED=disabled

cd /scratch/orybchuk/wakedynamics/ldm-3d/simulations/train

rm -rf post_processing
# cp /projects/wakedynamics/orybchuk/amr-wind2/amr-wind/build/amr_wind .
# cp /projects/wakedynamics/orybchuk/spack/amr-wind/spack-build-tamjn54/amr_wind .
# ln -sf /projects/awaken/orybchuk/spack-june22/amr-wind/spack-build-4ixvlaf/amr_wind .
ln -sf /projects/wakedynamics/orybchuk/spack-nov22/amr-wind/spack-build-4uqqmpu/amr_wind .

srun -n 144 -c 1 --cpu_bind=cores amr_wind abl_train.i
# srun -n 36 -c 1 --cpu_bind=cores amr_wind abl_spinup.i

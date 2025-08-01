#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --constraint=gpu&hbm80g
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --account=m3795_g
#SBATCH --qos=regular
#SBATCH --mem=0
#SBATCH --time=24:00:00
##SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mawright@lbl.gov

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=^docker0,lo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load conda
module load python
# module load nccl
# module load openmpi
conda activate me_121

srun python main.py \
system=slurm \
model/transformer=100querieswider \
dataset=smallgrid \
dataset.events_per_image_range=[1,2] \
training.batch_size=10 \
model.transformer.query_embeddings=100 \
model.denoising.use_denoising=true \
model.denoising.position_noise_std=5.0 \
model.denoising.mask_main_queries_from_denoising=true \
model.denoising.max_denoising_groups=10 \
training.tensorboard_name=1or2electrons_dn_all_losses \
$1

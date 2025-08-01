#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
##SBATCH --constraint=gpu&hbm80g
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --account=m3795_g
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mawright@lbl.gov

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=^docker0,lo
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE="1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export PYTORCH_JIT=0

module load conda
module load python
# module load nccl
# module load openmpi
conda activate me_121
srun python main.py \
system=slurm \
model/transformer=100querieswider \
dataset=smallgrid \
dataset.events_per_image_range=[1,1] \
dataset.pixels_file=pixelated_5um_tracks_thinned_4um_back_1M_300keV.txt \
model.transformer.query_embeddings=100 \
model.denoising.use_denoising=true \
model.denoising.mask_main_queries_from_denoising=true \
model.denoising.position_noise_std=5.0 \
model.denoising.max_denoising_groups=10 \
model.denoising.max_electrons_per_image=100 \
training.batch_size=10 \
training.tensorboard_name=1electron_dn_all_losses \
$1

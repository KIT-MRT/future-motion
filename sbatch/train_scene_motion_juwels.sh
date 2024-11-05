#!/bin/bash -x
#SBATCH --account=CHANGE_TO_YOUR_JUWELS_ACCOUNT
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --output=CHANGE_TO_YOUR_LOG_FOLDER/logs/slurm/scene_motion-out.%j
#SBATCH --error=CHANGE_TO_YOUR_LOG_FOLDER/logs/slurm/scene_motion-err.%j

# srun will no longer read in SLURM_CPUS_PER_TASK and will not inherit option
# --cpus-per-task from sbatch! This means you will explicitly have to specify
export SRUN_CPUS_PER_TASK=24

# Optional to disable the external environment, necessary, if python version is different
module purge

# Activate your conda env
source CHANGE_TO_YOUR_CONDA_PATH/miniconda3/bin/activate future-motion

# https://jugit.fz-juelich.de/aoberstrass/bda/ml-pipeline-template/-/blob/main/%7B%7Bcookiecutter.project_name%7D%7D/scripts/train_juwels.sbatch
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_CACHE_DIR="CHANGE_TO_YOUR_LOG_FOLDER/logs/wandb"

DATASET_DIR="CHANGE_TO_YOUR_DATA_FOLDER/data/waymo_motion" 
WANDB_ENTITY="CHANGE_TO_YOUR_WANDB_ACCOUNT"
WANDB_PROJECT="future-motion"
BATCH_SIZE=28 # A100: precision fp32
# BATCH_SIZE=32 # A6000 or A100 bf16
NUM_NODES=1
NUM_EPOCHS=50 # Joint version
# NUM_EPOCHS=128 # For training a marginal version from scratch 

# Cd to code and run
cd CHANGE_TO_YOUR_CODE_FOLDER/code/future-motion

srun python -u src/train_and_eval.py \
    trainer=womd \
    trainer.num_nodes=$NUM_NODES \
    trainer.max_epochs=$NUM_EPOCHS \
    model="ac_scene_motion" \
    datamodule.batch_size=$BATCH_SIZE \
    datamodule=h5_womd \
    loggers.wandb.name='train_scene_motion_${now:%Y-%m-%d-%H-%M-%S}' \
    loggers.wandb.project=$WANDB_PROJECT \
    loggers.wandb.entity=$WANDB_ENTITY \
    +loggers.wandb.offline=True \
    +logger.wandb.save_dir=$WANDB_CACHE_DIR \
    datamodule.data_dir=$DATASET_DIR \
    hydra.run.dir='CHANGE_TO_YOUR_LOG_FOLDER/logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'

wait
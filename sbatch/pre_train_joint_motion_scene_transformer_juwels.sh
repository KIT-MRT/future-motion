#!/bin/bash -x
#SBATCH --account=CHANGE_TO_YOUR_JUWELS_ACCOUNT
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=10:15:00
#SBATCH --output=CHANGE_TO_YOUR_LOG_FOLDER/logs/slurm/scene_transformer_joint_motion-out.%j
#SBATCH --error=CHANGE_TO_YOUR_LOG_FOLDER/logs/slurm/scene_transformer_joint_motion-err.%j

# srun will no longer read in SLURM_CPUS_PER_TASK and will not inherit option
# --cpus-per-task from sbatch! This means you will explicitly have to specify
export SRUN_CPUS_PER_TASK=12

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
BATCH_SIZE=12
NUM_NODES=1

# Cd to code and run
cd CHANGE_TO_YOUR_CODE_FOLDER/code/future-motion

srun python -u src/train_and_eval.py \
    trainer=womd \
    trainer.num_nodes=$NUM_NODES \
    +trainer.max_time="00:10:00:00" \
    model="sc_scene_transformer_joint_motion" \
    model.pre_training=True \
    +model.model.pre_training_mode="joint_motion" \
    datamodule.batch_size=$BATCH_SIZE \
    datamodule=h5_womd \
    loggers.wandb.name='pre_train_joint_motion_scene_transformer_${now:%Y-%m-%d-%H-%M-%S}' \
    loggers.wandb.project=$WANDB_PROJECT \
    loggers.wandb.entity=$WANDB_ENTITY \
    +loggers.wandb.offline=True \
    +logger.wandb.save_dir=$WANDB_CACHE_DIR \
    datamodule.data_dir=$DATASET_DIR \
    hydra.run.dir='CHANGE_TO_YOUR_LOG_FOLDER/logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'

wait
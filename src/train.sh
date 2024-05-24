DATASET_DIR="PATH_TO_DATASET" 
WANDB_ENTITY="WANDB_USER_OR_GROUP"
WANDB_PROJECT="WAND_PROJECT"
HYDRA_RUN_DIR="PATH_TO_HYDRALOGS"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

run () {
    python -u src/train.py \
    trainer=av2 \
    model=ac_red_motion \
    datamodule.batch_size=28 \
    datamodule=h5_womd \
    loggers.wandb.name='train_red_motion_${now:%Y-%m-%d-%H-%M-%S}' \
    loggers.wandb.project=$WANDB_PROJECT \
    loggers.wandb.entity=$WANDB_ENTITY \
    +logger.wandb.offline=True \
    datamodule.data_dir=$DATASET_DIR \
    hydra.run.dir='${HYDRA_RUNDIR}/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

run

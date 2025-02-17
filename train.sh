DATASET_DIR="PATH_TO_DATASET" 
WANDB_ENTITY="WANDB_USER_OR_GROUP"
WANDB_PROJECT="WANDB_PROJECT"
HYDRA_RUN_DIR="PATH_TO_HYDRALOGS"
BATCH_SIZE=28

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

run() {
    python -u future-motion/future_motion/train_and_eval.py \
    trainer=womd \
    model="$1" \
    datamodule.batch_size=$BATCH_SIZE \
    datamodule=h5_womd \
    loggers.wandb.name='train_'$1'_${now:%Y-%m-%d-%H-%M-%S}' \
    loggers.wandb.project=$WANDB_PROJECT \
    loggers.wandb.entity=$WANDB_ENTITY \
    +loggers.wandb.offline=True \
    datamodule.data_dir=$DATASET_DIR \
    hydra.run.dir=$HYDRA_RUN_DIR'/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

run $1

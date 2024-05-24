# FutureMotion
Common repo for our ongoing research on motion forecasting in self-driving vehicles.

## Setup
Clone this repo, afterwards init external submodules with:
```bash
git submodule update --init --recursive
```
Create conda environment named "future-motion" with:
```bash
conda env create -f conda_env.yml
```

Prepare Waymo Open Motion and Argoverse 2 Forecasting datasets by following the instructions in `src/external_submodules/hptr/README.md`.

## Motion forecasting models
<details>
<summary><big><b>RedMotion: Motion Prediction via Redundancy Reduction</b></big></summary>

TL;DR: Transformer model for motion prediction that incorporates two types of redundancy reduction.

<big><b>Overview</b></big>

![RedMotion](figures/red_motion.png "RedMotion")

Our model consists of two encoders. The trajectory encoder generates an embedding for the past trajectory of the current agent. The road environment encoder generates sets of local and global road environment embeddings as context. We use two redundancy reduction mechanisms, (a) architecture-induced and (b) self-supervised, to learn rich representations of road environments. All embeddings are fused via cross-attention to yield trajectory proposals per agent.

This repo contains the refactored implementation of RedMotion, the original implementation is available [here](https://github.com/kit-mrt/red-motion).

The Waymo Motion Prediction Challenge doesn't allow sharing the weights used in the challenge. However, we provide a [Colab notebook](https://colab.research.google.com/drive/16pwsmOTYdPpbNWf2nm1olXcx1ZmsXHB8) for a model with a shorter prediction horizon (5s vs. 8s) as a demo.

<big><b>Training</b></big>

To train a RedMotion model (tra-dec config) from scratch, adapt the global variables in train.sh according to your setup (Weights & Biases, local paths, batch size and visible GPUs).
The default batch size is set for A6000 GPUs with 48GB VRAM.
Then start the training run with:
```bash
bash train.sh ac_red_motion
```

</details>


## Acknowledgements
This repo builds upon the great work [HPTR](https://github.com/zhejz/HPTR) by [@zhejz](https://github.com/zhejz). 

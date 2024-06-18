# FutureMotion
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning) ![Black](https://img.shields.io/badge/code%20style-black-000000.svg) ![Weights & Biases](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)

Common repo for our ongoing research on motion forecasting in self-driving vehicles.

## Setup
Clone this repo, afterwards init external submodules with:
```bash
git submodule update --init --recursive
```
Create a conda environment named "future-motion" with:
```bash
conda env create -f conda_env.yml
```

Prepare Waymo Open Motion and Argoverse 2 Forecasting datasets by following the instructions in `src/external_submodules/hptr/README.md`.

## Methods

### RedMotion: Motion Prediction via Redundancy Reduction 

![RedMotion](figures/red_motion.png "RedMotion")

Our RedMotion model consists of two encoders. The trajectory encoder generates an embedding for the past trajectory of the current agent. The road environment encoder generates sets of local and global road environment embeddings as context. We use two redundancy reduction mechanisms, (a) architecture-induced and (b) self-supervised, to learn rich representations of road environments. All embeddings are fused via cross-attention to yield trajectory proposals per agent.

<details>
<summary><big><b>More details</b></big></summary>

This repo contains the refactored implementation of RedMotion, the original implementation is available [here](https://github.com/kit-mrt/red-motion).

The Waymo Motion Prediction Challenge doesn't allow sharing the weights used in the challenge. However, we provide a [Colab notebook](https://colab.research.google.com/drive/16pwsmOTYdPpbNWf2nm1olXcx1ZmsXHB8) for a model with a shorter prediction horizon (5s vs. 8s) as a demo.

<big><b>Training</b></big>

To train a RedMotion model (tra-dec config) from scratch, adapt the global variables in train.sh according to your setup (Weights & Biases, local paths, batch size and visible GPUs).
The default batch size is set for A6000 GPUs with 48GB VRAM.
Then start the training run with:
```bash
bash train.sh ac_red_motion
```
For reference, this [wandb plot](https://wandb.ai/kit-mrt/red-motion-hptr/reports/waymo_pred-mean_average_precision-24-05-25-17-50-52---Vmlldzo4MDkyMjQ2?accessToken=j7a8pf4wvm9g6gvy95f88h0asdy57few6rw1jvv1qrf9jzuwpnirzv975id3pgxn) shows the validation mAP scores for the epochs 23 - 129 (default config, trained on 4 A6000 GPUs for ~100h).

<big><b>Reference</b></big>
```bibtex
@article{
    wagner2024redmotion,
    title={RedMotion: Motion Prediction via Redundancy Reduction},
    author={Royden Wagner and Omer Sahin Tas and Marvin Klemp and Carlos Fernandez and Christoph Stiller},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
}
```

</details>

### Words in Motion: Representation Engineering for Motion Forecasting

![Words in Motion](figures/words_in_motion.png "Words in Motion")

We use natural language to quantize motion features in an inter-pretable way. (b) The corresponding direction, speed, and acceleration classes are highlighted in blue. (c) To reverse engineer motion forecasting models, we measure the degree to which these features are embedded in their hidden states H with linear probes. Furthermore, we use our discrete motion features to fit control vectors V that allow for controlling motion forecasts during inference.

<details>
<summary><big><b>More details</b></big></summary>

<big><b>Gradio demos</b></big>

Use [this Colab notebook](https://colab.research.google.com/drive/1ItY9YWQAmpfwc8KTRp6oY9e4uUWKxZrX?usp=sharing) to start Gradio demos for our speed control vectors.

In contrast to the qualitative results in our paper, we show the motion forecasts for the focal agent and 8 other agents in a scene. 
Press the submit button with the default temperature = 0 to visualize the default (non-controlled) forecasts, then change the temperature and resubmit to visualize the changes. 
The example is from the Waymo Open dataset and shows motion forecasts for vehicles and a pedestrian (top center).

<big><b>Training</b></big>

Soon to be released.

</details>


## Acknowledgements
This repo builds upon the great work [HPTR](https://github.com/zhejz/HPTR) by [@zhejz](https://github.com/zhejz). 

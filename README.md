# Nano T2I Diffusion on [MONET](https://huggingface.co/datasets/jasperai/monet) Dataset 
This repository contains the code to train a Text-to-Image Diffusion Model on the [MONET](https://huggingface.co/datasets/jasperai/monet) dataset.

<figure>
	<p align="center">
        	<img style="width:400px;" src="assets/monet.jpg">
	 </p>
</figure>

## Setup

To be up and running, you need first to create a virtual env with `python >=3.12` installed and activate it.

### With `virtualenv`

```bash
python3.13 -m virtualenv envs/nano
source envs/nano/bin/activate
```

### With `conda`

```bash
conda create -n nano python=3.13
conda activate nano
```

Then install the required dependencies (if on GPU) and the repo in editable mode

```bash
pip install --upgrade pip
pip install -e ".[training]" # for training
```

## Train the model

We provide config files for the trainings in `examples/trainings/configs` as well as a script to train the model in `examples/trainings/training.py`. In particular, we provide a config to train a **nano** model (trainable on a single H100 GPU) in `examples/trainings/configs/nano.yaml`.

To train the model, you can run the following command:

```bash
python examples/trainings/training.py examples/trainings/configs/t2i/nano.yaml
```

Once the training is launched, you can visualize the training progress on [wandb](https://wandb.ai). Checkpoints will be saved in the `examples/trainings/nano-t2i/checkpoints` directory.

## Demo

You can find a gradio demo in `examples/inference/demo/t2i_demo.py` allowing you to generate images from the trained model. In this demo, you can select the model config file and the checkpoint name to use.

To run the demo, you can run the following command:

```bash
python examples/inference/demo/t2i_demo.py
```

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{aubin2026monet,
  title   = {MONET: A Massive, Open, Non-redundant and Enriched Text-to-image Dataset},
  author  = {Aubin, Benjamin and Quintana, Gonzalo I{\~n}aki and Tasar, Onur and Sreetharan, Sanjeev and Czerwinska, Urszula and Henry, Damien and Chadebec, Cl{\'e}ment},
  year    = {2026},
  note    = {Jasper Research}
}
```
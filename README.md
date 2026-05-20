<p align="center">
  <img src="assets/logo.svg" alt="nano-t2i" width="560"/>
</p>

<figure>
	<p align="center">
        	<img style="width:400px;" src="assets/monet.jpg">
	 </p>
</figure>

**A minimal, hackable codebase to train a text-to-image (T2I) flow-matching model end-to-end on [MONET dataset](https://huggingface.co/datasets/jasperai/monet) and a single GPU**. Trains a T2I model from scratch on 1×H200 under 300$. In the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat), but for diffusion & flow-matching.


## Setup

To be up and running, you need first to create a virtual env with `python >=3.12` installed and activate it.

### With `uv`

```shell
uv venv envs/nano-t2i --python 3.13
source envs/nano-t2i/bin/activate
uv pip install -e ".[training]"
```

### With `virtualenv`

```shell
python3.13 -m virtualenv envs/nano-t2i
source envs/nano-t2i/bin/activate
pip install --upgrade pip
pip install -e ".[training]"
```

### With `conda`

```shell
conda create -n nano-t2i python=3.13
conda activate nano-t2i
pip install --upgrade pip
pip install -e ".[training]"
```

## Train the model

We provide config files for the trainings in `examples/trainings/configs` as well as a script to train the model in `examples/trainings/training.py`. In particular, we provide a config to train a **nano** model (trainable on a single H100 GPU) in `examples/trainings/configs/nano.yaml`.

To train the model, you can run the following command:

```shell
python examples/trainings/training.py examples/trainings/configs/nano.yaml
```

Once the training is launched, you can visualize the training progress on [wandb](https://wandb.ai). Checkpoints will be saved in the `examples/trainings/nano-t2i/checkpoints` directory.

## Results

Below is an example of the training progress for two training runs 1) on a single H200 GPU and 2) on a 8 H200 GPUs.


<figure>
	<p align="center">
        	<img style="width:400px;" src="assets/training_curves.jpg">
	 </p>
</figure>

| Resolution | Hardware | Wall time | Cost (@ $3/H200/h) | Examples after 1 day of training |
|---|---|---|---|---|
| 512  | 1×H200 | 24 h | ~\$72  |  <img src="assets/gen_single_1_day_3.jpg" width="100"> <img src="assets/gen_single_1_day_1.jpg" width="100"> <img src="assets/gen_single_1_day_2.jpg" width="100"> <img src="assets/gen_single_1_day_0.jpg" width="100"> |
| 512  | 1×H200 | 36 h  | ~\$108  | |
| 1024 | 1×H200 | 48 h  | ~\$144  | |
| 1024 | 1×H200 | 72 h  | ~\$216  | |
| 1024 | 1×H200 | 96 h  | ~\$288  | |

## Demo

You can find a gradio demo in `examples/inference/demo/t2i_demo.py` allowing you to generate images from the trained model. In this demo, you can select the model config file and the checkpoint name to use.

To run the demo, you can run the following command:

```shell
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

## TODOS

[ ] Add Flash-attn 3 install
[ ] Change print to log.debug
```shell
	INFO:root:loss: 1.3965709209442139, global_rank:0, local_rank:0
	INFO:root:END training_step
	INFO:root:on_after_backward
	INFO:root:START on_train_batch_end
	INFO:root:on_train_batch_end
	INFO:root:END on_train_batch_end
	--------------------------------
	Time to get VAE embedding 0.00010323303285986185
	Time to get conditioning 0.06963227200321853
	Time to sample timestep 0.044684075051918626
	Time to predict noise 0.018585636978968978
	Time to compute latent loss 0.0001023260410875082
	out: {'loss': tensor(1.7208, device='cuda:0', grad_fn=<MeanBackward0>), 'latent_loss': tensor(1.7208, device='cuda:0', grad_fn=<MeanBackward0>)}
	Forward time: 0.24001224397215992
```
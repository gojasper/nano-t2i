# nano-t2i

<p align="center">
  <img src="assets/logo_nano_t2i.png" alt="nano-t2i" width="560"/>
</p>

**A minimal, hackable, open codebase to train reproducibly a text-to-image (T2I) flow-matching model end-to-end on the [MONET dataset (Apache-2.0)](https://huggingface.co/datasets/jasperai/monet) — on a single H200 GPU, under \$300.**
<p align="center">
  <a href="https://arxiv.org/abs/2605.21272"><img src="https://img.shields.io/badge/arXiv-2605.21272-b31b1b.svg?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/jasperai/monet"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MONET-ffcc4d" alt="MONET Dataset"></a>
  <a href="https://huggingface.co/spaces/jasperai/monet-retrieval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-MONET%20Retrieval-ffcc4d" alt="MONET Retrieval"></a>
  <a href="https://huggingface.co/spaces/jasperai/monet-umap"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-MONET%20UMAP-ffcc4d" alt="MONET UMAP"></a>
</p>

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Demo](#demo)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Overview

`nano-t2i` is a small 1.3B DiT-style **flow-matching** text-to-image model with a Qwen3-4B text encoder and a latent VAE backbone, trained in two phases (512 → 1024) on the [MONET](https://huggingface.co/datasets/jasperai/monet) synthetic captioned-image dataset. The model relies on AdaLN sharing and is initialized using AdaLN-Zero. It is built on top of [PyTorch Lightning](https://lightning.ai/) and [diffusers](https://github.com/huggingface/diffusers), and is designed to be:

- **Small enough to fit on a single H200 GPU** (the `nano` config: 5 dual-stream + 5 single-stream DiT blocks, 24 attention heads, 128-dim heads with AdaLN sharing).
- **Hackable**: every architectural choice lives in the YAML config (see [`examples/trainings/configs/nano.yaml`](examples/trainings/configs/nano.yaml)).
- **End-to-end reproducible**: from [MONET](https://huggingface.co/datasets/jasperai/monet) shards to a working Gradio demo, in two commands.

*Note:* This codebase also supports `flash attention` v3. We refer to [flash-attn repo](https://github.com/Dao-AILab/flash-attention/) for proper installation in your own environment.

## Results

The figure below shows the training progress for two reference runs: 1×H200 and 8×H200.

<p align="center">
  <img src="assets/training_curves.jpg" width="600" alt="nano-t2i training loss"/>
</p>

*Note:* This codebase can be used to train bigger models by modifying the training configs and/or the code itself. In particular, it was used to train a 4B model the results of which are available in [our paper](https://arxiv.org/abs/2605.21272).


### Runs

Cost is computed at **~\$3 / H200 / hour** (representative of major cloud GPU providers; check your own pricing). Click a thumbnail to open the full-resolution image.

| Resolution | Hardware | Wall time | Cost  | Example samples |
|---|---|---|---|---|
| 512  | 1×H200 | 24 h | ~\$72  | <a href="assets/24h/3.jpg"><img src="assets/24h/3.jpg" width="60"></a> <a href="assets/24h/1.jpg"><img src="assets/24h/1.jpg" width="60"></a> <a href="assets/24h/2.jpg"><img src="assets/24h/2.jpg" width="60"></a> <a href="assets/24h/0.jpg"><img src="assets/24h/0.jpg" width="60"></a><a href="assets/24h/4.jpg"><img src="assets/24h/4.jpg" width="60"></a><a href="assets/24h/5.jpg"><img src="assets/24h/5.jpg" width="60"></a> |
| 512  | 1×H200 | 36 h | ~\$108 | <a href="assets/36h/3.jpg"><img src="assets/36h/3.jpg" width="60"></a> <a href="assets/36h/1.jpg"><img src="assets/36h/1.jpg" width="60"></a> <a href="assets/36h/2.jpg"><img src="assets/36h/2.jpg" width="60"></a> <a href="assets/36h/0.jpg"><img src="assets/36h/0.jpg" width="60"></a><a href="assets/24h/4.jpg"><img src="assets/36h/4.jpg" width="60"></a><a href="assets/36h/5.jpg"><img src="assets/36h/5.jpg" width="60"></a>  |
| 1024 | 1×H200 | 48 h | ~\$144 | <a href="assets/48h/3.jpg"><img src="assets/48h/3.jpg" width="60"></a> <a href="assets/48h/1.jpg"><img src="assets/48h/1.jpg" width="60"></a> <a href="assets/48h/2.jpg"><img src="assets/48h/2.jpg" width="60"></a> <a href="assets/48h/0.jpg"><img src="assets/48h/0.jpg" width="60"></a><a href="assets/24h/4.jpg"><img src="assets/48h/4.jpg" width="60"></a><a href="assets/48h/5.jpg"><img src="assets/48h/5.jpg" width="60"></a>  |
| 1024 | 1×H200 | 60 h | ~\$180 | <a href="assets/60h/3.jpg"><img src="assets/60h/3.jpg" width="60"></a> <a href="assets/60h/1.jpg"><img src="assets/60h/1.jpg" width="60"></a> <a href="assets/60h/2.jpg"><img src="assets/60h/2.jpg" width="60"></a> <a href="assets/60h/0.jpg"><img src="assets/60h/0.jpg" width="60"></a><a href="assets/60h/4.jpg"><img src="assets/60h/4.jpg" width="60"></a><a href="assets/60h/5.jpg"><img src="assets/60h/5.jpg" width="60"></a>  |
| 1024 | 1×H200 | 72 h | ~\$216 | <a href="assets/72h/3.jpg"><img src="assets/72h/3.jpg" width="60"></a> <a href="assets/72h/1.jpg"><img src="assets/72h/1.jpg" width="60"></a> <a href="assets/72h/2.jpg"><img src="assets/72h/2.jpg" width="60"></a> <a href="assets/72h/0.jpg"><img src="assets/72h/0.jpg" width="60"></a><a href="assets/72h/4.jpg"><img src="assets/72h/4.jpg" width="60"></a><a href="assets/72h/5.jpg"><img src="assets/72h/5.jpg" width="60"></a>  |

**Note:** You may need to input long prompts to generate nice images at the begginging of the training.

## Setup

You need Python `>=3.13` and a CUDA 12.8-compatible driver (PyTorch 2.9 / cu128 wheels are pinned in [`requirements.txt`](requirements.txt)).

Clone the repo first:

```shell
git clone https://github.com/gojasper/nano-t2i.git
cd nano-t2i
```

### With `uv`

```shell
uv venv envs/nano-t2i --python 3.13
source envs/nano-t2i/bin/activate
uv pip install -e ".[training]"
```

<details>
<summary>Alternative install paths (<code>virtualenv</code>, <code>conda</code>)</summary>

#### With `virtualenv`

```shell
python3.13 -m virtualenv envs/nano-t2i
source envs/nano-t2i/bin/activate
pip install --upgrade pip
pip install -e ".[training]"
```

#### With `conda`

```shell
conda create -n nano-t2i python=3.13
conda activate nano-t2i
pip install --upgrade pip
pip install -e ".[training]"
```

</details>

## Dataset

`nano-t2i` trains on [**MONET**](https://huggingface.co/datasets/jasperai/monet) (Massive, Open, Non-redundant and Enriched Text-to-image dataset), a curated **104.9M** image-text corpus distilled from 2.9B raw pairs across nine open sources (six real, three synthetic) with safety filtering, deduplication (pHash + SSCD), domain governance, and multi-VLM re-captioning. MONET is released under **Apache-2.0** and ships pre-computed SANA-VAE latents for direct latent-diffusion training. See the [dataset card](https://huggingface.co/datasets/jasperai/monet) for full details.

## Training

Training configs live in [`examples/trainings/configs/`](examples/trainings/configs); the main entrypoint is [`examples/trainings/training.py`](examples/trainings/training.py). The reference config is [`nano.yaml`](examples/trainings/configs/nano.yaml), which defines two sequential phases:

1. `nano-512` — 200k steps at 512×512.
2. `nano-1024` — 500k steps at 1024×1024, resumed from phase 1.

```shell
python examples/trainings/training.py --path_config examples/trainings/configs/nano.yaml
```

### Logging with Weights & Biases

Training metrics are logged to W&B under the `Nano-T2I` project (configurable via `logging.wandb_project` in the YAML). To authenticate:

```shell
# interactive
wandb login
# or
export WANDB_API_KEY=...
```

### Checkpoints

Checkpoints are written to the path specified by `training.save_ckpt_path` in each phase of the config (default: `logs/nano/phase-1/` and `logs/nano/phase-2/`). Phase 2 resumes from `logs/nano/phase-1/last.ckpt` by default — make sure phase 1 has completed before launching it.

## Demo

A Gradio demo is provided in [`examples/inference/demo/t2i_demo.py`](examples/inference/demo/t2i_demo.py). It lets you pick a config file and a checkpoint name, and generate images interactively.

```shell
python examples/inference/demo/t2i_demo.py
```

## Contributing

We welcome bug reports, documentation improvements, config updates, and focused code changes. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and the pull request process before opening a PR.
## Acknowledgements

Built on the shoulders of [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/), [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers), [Qwen3](https://huggingface.co/Qwen), and the [SANA](https://github.com/NVlabs/Sana) VAE. The MONET dataset and reference runs are released by [Jasper Research](https://huggingface.co/jasperai).

## Citation

If you use this code or the MONET dataset in your research, please cite:

```bibtex
@article{aubin2026monet,
        title   = {MONET: A Massive, Open, Non-redundant and Enriched Text-to-image Dataset},
        author  = {Aubin, Benjamin and Quintana, Gonzalo I{\~n}aki and Tasar, Onur and Sreetharan, Sanjeev and Czerwinska, Urszula and Henry, Damien and Chadebec, Cl{\'e}ment},
        journal=  {arXiv preprint arXiv:2605.21272},
        year    = {2026},
        note    = {Jasper Research}
}
```

## License

This codebase is released under the [Apache 2.0 License](LICENSE). The MONET dataset has its own license — please consult the [dataset card](https://huggingface.co/datasets/jasperai/monet) before redistributing.


---
> Curious how Jasper Research is used in production?  
> <a href="https://developers.jasper.ai/docs/using-images?utm_source=JResearch&amp;utm_medium=CTA&amp;utm_campaign=MONET" target="_blank" rel="noopener noreferrer" aria-label="Discover Jasper APIs for image workflows">Discover Jasper APIs for image workflows</a>
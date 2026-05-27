import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from nano_t2i.models.diffusion import DiffusionModel, DiffusionModelConfig
from nano_t2i.models.embedders import (
    ConditionerWrapper,
    QwenEmbedder,
    QwenEmbedderConfig,
)
from nano_t2i.models.transformers.tranformers import FluxTransformer
from nano_t2i.models.vae import AutoencoderDCDiffusers, AutoencoderDCDiffusersConfig

ProgressFn = Callable[[float, str], None]


def _noop_progress(_frac: float, _desc: str) -> None:
    pass


def _build_denoiser(
    patch_size: int,
    num_layers: int,
    num_single_layers: int,
    attention_head_dim: int,
    num_attention_heads: int,
    context_dim: int,
    axes_dims_rope: List[int],
    mlp_ratio: float,
    use_context_rms_norm: bool,
    share_modulation: bool,
    adaln_zero_init: bool,
    use_adaln_learnable_embedding: bool,
    base_resolution: Optional[Tuple[int, int]],
) -> FluxTransformer:
    denoiser_config = {
        "patch_size": patch_size,
        "in_channels": 32 * (patch_size**2),
        "out_channels": 32 * (patch_size**2),
        "num_layers": num_layers,
        "num_single_layers": num_single_layers,
        "attention_head_dim": attention_head_dim,
        "num_attention_heads": num_attention_heads,
        "context_dim": context_dim,
        "axes_dims_rope": axes_dims_rope,
        "rope_theta": 10_000,
        "mlp_ratio": mlp_ratio,
        "qkv_bias": True,
        "use_context_rms_norm": use_context_rms_norm,
        "share_modulation": share_modulation,
        "adaln_zero_init": adaln_zero_init,
        "use_adaln_learnable_embedding": use_adaln_learnable_embedding,
        "base_resolution": base_resolution,
    }
    denoiser = FluxTransformer(**denoiser_config).to(torch.bfloat16)
    num_params = sum(p.numel() for p in denoiser.parameters())
    logging.info(f"Number of denoiser parameters: {num_params}")
    return denoiser


def _build_text_embedder(
    text_embedder: str,
    tokenizer_max_length: int,
    tokenizer_padding: str,
) -> QwenEmbedder:
    if text_embedder != "qwen3-4b":
        raise ValueError(f"Text embedder {text_embedder} not supported")

    text_embedder_config = QwenEmbedderConfig(
        version="Qwen/Qwen3-4B-Instruct-2507",
        text_embedder_subfolder="",
        tokenizer_subfolder="",
        returns_attention_mask=True,
        unconditional_conditioning_value="",
        tokenizer_max_length=tokenizer_max_length,
        tokenizer_padding=tokenizer_padding,
        layer_idx=-2,
        unconditional_conditioning_rate=0.1,
    )
    embedder = QwenEmbedder(text_embedder_config).to(torch.bfloat16)
    embedder.freeze()
    return embedder


def _build_vae() -> AutoencoderDCDiffusers:
    vae_config = AutoencoderDCDiffusersConfig(
        version="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        subfolder="vae",
        tiling_size=(128, 128),
    )
    return AutoencoderDCDiffusers(vae_config).to(torch.bfloat16)


def get_model(
    timestep_sampling: str = "log_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    prediction_type: str = "flow_matching",
    tokenizer_max_length: int = 256,
    tokenizer_padding: str = "longest",
    num_layers: int = 5,
    num_single_layers: int = 5,
    patch_size: int = 1,
    num_attention_heads: int = 24,
    attention_head_dim: int = 128,
    mlp_ratio: float = 2.0,
    axes_dims_rope: List[int] = [16, 56, 56],
    use_context_rms_norm: bool = False,
    share_modulation: bool = False,
    use_adaln_learnable_embedding: bool = False,
    adaln_zero_init: bool = False,
    text_embedder: str = "qwen3-4b",
    vae_input_key: str = "latent",
    conditioner_input_key: str = "text",
    base_resolution: Optional[Tuple[int, int]] = None,
    progress_cb: ProgressFn = _noop_progress,
):
    if text_embedder == "qwen3-4b":
        context_dim = 2560
    else:
        raise ValueError(f"Text embedder {text_embedder} not supported")

    progress_cb(0.02, "Building denoiser…")
    denoiser = _build_denoiser(
        patch_size=patch_size,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        context_dim=context_dim,
        axes_dims_rope=axes_dims_rope,
        mlp_ratio=mlp_ratio,
        use_context_rms_norm=use_context_rms_norm,
        share_modulation=share_modulation,
        adaln_zero_init=adaln_zero_init,
        use_adaln_learnable_embedding=use_adaln_learnable_embedding,
        base_resolution=base_resolution,
    )

    progress_cb(0.10, "Loading text encoder (Qwen3-4B)…")
    embedder = _build_text_embedder(
        text_embedder=text_embedder,
        tokenizer_max_length=tokenizer_max_length,
        tokenizer_padding=tokenizer_padding,
    )
    conditioner = ConditionerWrapper(conditioners=[embedder]).to(torch.bfloat16)

    progress_cb(0.55, "Loading VAE (SANA1.5)…")
    vae = _build_vae()

    progress_cb(0.75, "Assembling diffusion model…")
    config = DiffusionModelConfig(
        input_key=vae_input_key,
        ucg_keys=[conditioner_input_key],
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        prediction_type=prediction_type,
    )
    model = DiffusionModel(
        config,
        denoiser=denoiser,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)
    return model


def load_ckpt(
    path_ckpt: str,
    model: DiffusionModel,
    ckpt_name: str = "last.ckpt",
):

    ckpt_path = os.path.join(path_ckpt, ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        global_step = ckpt["global_step"]
        state_dict = ckpt["state_dict"]
        sd = {k[6:]: v for k, v in state_dict.items() if "model." in k}
        model.load_state_dict(sd, strict=False)
        logging.info(f"Model loaded from checkpoint: {ckpt_path}")
        return model, global_step
    else:
        raise FileNotFoundError(f"Checkpoint {path_ckpt} not found")


@torch.no_grad()
def sample_from_batch(
    model: DiffusionModel,
    batch: Dict[str, Any],
    num_steps: int,
    guidance_scale: float,
):
    logs = model.log_samples(
        batch=batch,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        max_samples=len(batch[model.input_key]),
        cfg_normalization=True,
    )

    return logs


def get_model_from_config(
    config: Dict[str, Any],
    ckpt_name: str = "last.ckpt",
    ckpt_path: str = "checkpoints",
    progress_cb: ProgressFn = _noop_progress,
):
    model = get_model(**config, progress_cb=progress_cb)
    progress_cb(0.85, "Loading checkpoint weights…")
    model, global_step = load_ckpt(
        path_ckpt=ckpt_path, model=model, ckpt_name=ckpt_name
    )
    return model, global_step

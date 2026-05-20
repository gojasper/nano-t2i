import datetime
import logging
import os
import random
import re
import uuid
from typing import List, Optional, Tuple

import braceexpand
import fire
import PIL
import torch
import webdataset as wds
import yaml
from bloom_mappers import generic_mappers
from nano_t2i.data.datasets import (
    DataModuleConfig,
    MultiDataModule,
    MultiDataModuleConfig,
)
from nano_t2i.data.mappers import RescaleMapper, RescaleMapperConfig
from nano_t2i.data.mappers_batched import (
    MultiAspectRatioCacherConfig,
    MultiAspectRatioCacherMapper,
)
from nano_t2i.models.diffusion import DiffusionModel, DiffusionModelConfig
from nano_t2i.models.embedders import (
    ConditionerWrapper,
    Gemma3Embedder,
    Gemma3EmbedderConfig,
    Gemma3EmbeddingEmbedder,
    Gemma3EmbeddingEmbedderConfig,
    GemmaEmbedder,
    GemmaEmbedderConfig,
    IdentityEmbedder,
    IdentityEmbedderConfig,
    QwenEmbedder,
    QwenEmbedderConfig,
    T5TextEmbedder,
    T5TextEmbedderConfig,
)
from nano_t2i.models.transformers.flux_utils import (
    DoubleStreamBlock,
    Modulation,
    SingleStreamBlock,
)
from nano_t2i.models.transformers.tranformers import FluxTransformer
from nano_t2i.models.vae import (
    AutoencoderDCDiffusers,
    AutoencoderDCDiffusersConfig,
    AutoencoderKLDiffusers,
    AutoencoderKLDiffusersConfig,
)
from nano_t2i.trainer import TrainingConfig, TrainingPipeline
from nano_t2i.trainer.loggers import WandbSampleLogger
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import Compose, Normalize, Resize
from transformers import AutoImageProcessor, AutoModel

PIL.Image.MAX_IMAGE_PIXELS = 933120000

MAPPER_BLOOM = {
    "diffusion-aesthetic-4k": generic_mappers,
    "cc12m": generic_mappers,
    "rosemary": generic_mappers,
    "imagenet-1k": generic_mappers,
    "multigen20m": generic_mappers,
    "megalith10m": generic_mappers,
    "tulip": generic_mappers,
    "synthetic_text_to_image": generic_mappers,
    "commoncatalog": generic_mappers,
    "pexels": generic_mappers,
    "unsplash": generic_mappers,
}


def get_dataset_configs_from_config(config: dict):
    configs = []
    pre_filters_mappers = []

    mappers = config.get("mappers", [])

    global_batch_sizes = config.get("global_batch_sizes")
    global_pixel_budgets = config.get("global_pixel_budgets")
    global_aspect_ratios = config.get("global_aspect_ratios")
    global_intervals = config.get("global_intervals")
    global_probabilities = config.get("global_probabilities")
    shuffle_shards = config.get("shuffle_shards", True)
    decoder = config.get("decoder", "pil")
    tokenizer_max_length = config.get("tokenizer_max_length", None)
    print(f"Tokenizer max length: {tokenizer_max_length}")

    dataset_sampling_probabilities = []

    for mapper in mappers:
        mapper_name = mapper.get("name")
        dataset_sampling_probabilities.append(
            mapper.get("probability", 1.0 / len(mappers))
        )

        if mapper_name not in MAPPER_BLOOM:
            raise ValueError(f"Mapper name {mapper_name} not found in MAPPER_BLOOM")

        shards = mapper.get("shards", [])
        bucket = mapper.get("bucket", "jasper-ai-research")
        prefix = mapper.get("prefix", "pipe:hfcli buckets cp hf://buckets/jasperai/")
        if isinstance(shards, str):
            shards = [shards]

        shards_path_or_urls_unbraced = []
        for shard in shards:
            all_tar_files = list(braceexpand.braceexpand(f"{prefix}{bucket}/{shard} -"))
            logging.info(
                f"Collected {len(all_tar_files)} tar files for dataset {mapper_name}"
            )
            shards_path_or_urls_unbraced.extend(all_tar_files)

        # shuffle shards
        if shuffle_shards:
            random.shuffle(shards_path_or_urls_unbraced)
            logging.info("-" * 100)
            logging.info(f"SHUFFLED SHARDS for dataset {mapper_name}")
        logging.info(shards_path_or_urls_unbraced[:10])
        logging.info(f"Number of shards: {len(shards_path_or_urls_unbraced)}")
        logging.info("-" * 100)

        mapper_kwargs = mapper.get("mapper_kwargs", {})
        mappers = MAPPER_BLOOM[mapper_name](
            **mapper_kwargs, handles_image=decoder is not None
        )
        pre_filters_mappers.append(mappers)

        configs.append(
            DataModuleConfig(
                shards_path_or_urls=shards_path_or_urls_unbraced,
                decoder=decoder,
                shuffle_before_split_by_node_buffer_size=mapper.get(
                    "shuffle_before_split_by_node_buffer_size", 1
                ),
                shuffle_before_split_by_workers_buffer_size=mapper.get(
                    "shuffle_before_split_by_workers_buffer_size", 1
                ),
                shuffle_before_filter_mappers_buffer_size=mapper.get(
                    "shuffle_before_filter_mappers_buffer_size", 10
                ),
                shuffle_after_filter_mappers_buffer_size=mapper.get(
                    "shuffle_after_filter_mappers_buffer_size", 10
                ),
                per_worker_batch_size=1,
                num_workers=mapper.get("num_workers", 4),
                # handler=wds.reraise_exception,
            )
        )

    if decoder is None:
        batched_mapper = None

    else:
        batched_mapper = MultiAspectRatioCacherMapper(
            MultiAspectRatioCacherConfig(
                input_keys=[
                    "image",
                ],
                output_keys=[
                    "image",
                ],
                batch_sizes=global_batch_sizes,
                probabilities=global_probabilities,
                pixel_budgets=global_pixel_budgets,
                aspect_ratios=global_aspect_ratios,
                intervals=global_intervals,
                filters_mappers=[
                    RescaleMapper(RescaleMapperConfig(key="image")),
                ],
            ),
        )

    multi_data_module_config = MultiDataModuleConfig(
        configs=configs,
        dataset_sampling_probabilities=dataset_sampling_probabilities,
        per_worker_batch_size=global_batch_sizes[0] if batched_mapper is None else 1,
        max_length=tokenizer_max_length,
    )

    return multi_data_module_config, pre_filters_mappers, batched_mapper


def get_data_module_from_config(config: dict):
    train_config = config.get("train", None)
    validation_config = config.get("validation", None)

    train_multi_data_module_config, train_pre_filters_mappers, train_batched_mapper = (
        get_dataset_configs_from_config(train_config)
    )

    if validation_config:
        (
            validation_multi_data_module_config,
            validation_pre_filters_mappers,
            validation_batched_mapper,
        ) = get_dataset_configs_from_config(validation_config)
    else:
        validation_multi_data_module_config = None
        validation_pre_filters_mappers = None
        validation_batched_mapper = None
    data_module = MultiDataModule(
        train_config=train_multi_data_module_config,
        train_filters_mappers=train_pre_filters_mappers,
        train_batched_filters_mappers=train_batched_mapper,
        eval_config=validation_multi_data_module_config,
        eval_filters_mappers=validation_pre_filters_mappers,
        eval_batched_filters_mappers=validation_batched_mapper,
    )
    return data_module


def get_model(
    timestep_sampling: str = "log_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    logit_mean_schedule: List[float] = None,
    logit_std_schedule: List[float] = None,
    timestep_change_steps: List[int] = None,
    prediction_type: str = "flow_matching",
    tokenizer_max_length: int = 77,
    tokenizer_padding: str = "max_length",
    use_ema: bool = False,
    ema_decay: float = 0.9999,
    ema_update_every_n_steps: int = 50,
    start_ema_after_n_steps: int = 5000,
    num_layers: int = 8,
    num_single_layers: int = 8,
    attn_type: str = "flash_attn_v3",
    vae_name: str = "FLUX.1-schnell",
    patch_size: int = 2,
    num_attention_heads: int = 24,
    attention_head_dim: int = 128,
    mlp_ratio: float = 4.0,
    use_batch_norm: bool = False,
    axes_dims_rope: List[int] = [16, 56, 56],
    use_context_rms_norm: bool = False,
    use_pooled_projection_rms_norm: bool = False,
    use_text_attention_mask: bool = False,
    use_text_positional_encoding: bool = False,
    share_modulation: bool = False,
    use_adaln_learnable_embedding: bool = False,
    adaln_zero_init: bool = False,
    text_embedder: str = "t5",
    pooled_text_embedder: Optional[str] = None,
    vae_input_key: str = "image",
    use_timestep_shifting: bool = False,
    base_resolution: Optional[Tuple[int, int]] = None,
    conditioner_input_key: str = "text",
    repa_loss_weight: float = 0.5,
    use_irepa: bool = False,
    repa_model: str = "dinov2_vitg14",
    repa_projectors_target_layers: List[int] = [8],
    repa_projectors_hidden_dim: int = 2048,
):
    if vae_name == "SANA":
        vae_channels = 32
    elif vae_name == "FLUX.1-schnell":
        vae_channels = 16
    elif vae_name == "FLUX.2-dev":
        vae_channels = 32
    else:
        raise ValueError(f"VAE name {vae_name} not supported")

    if use_irepa:
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        if repa_model == "dinov2-giant":
            repa_projectors_z_dim = 1536
        elif repa_model == "dinov2-base":
            repa_projectors_z_dim = 1024
        elif repa_model == "dinov2-large":
            repa_projectors_z_dim = 768
        elif repa_model == "dinov2-small":
            repa_projectors_z_dim = 384
        elif repa_model == "dinov3-vitl16-pretrain-lvd1689m":
            repa_projectors_z_dim = 1024
        elif repa_model == "dinov3-vit7b16-pretrain-lvd1689m":
            repa_projectors_z_dim = 4096
        else:
            raise ValueError(f"Repa model {repa_model} not supported")

        repa_encoder_image_processor = AutoImageProcessor.from_pretrained(
            f"facebook/{repa_model}",
        )
        repa_encoder = AutoModel.from_pretrained(
            f"facebook/{repa_model}",
            device_map="cpu",
        ).to(torch.bfloat16)
        repa_encoder.eval()
    else:
        repa_encoder = None
        repa_encoder_image_processor = None
        repa_projectors_z_dim = None
        repa_projectors_target_layers = None
        repa_projectors_hidden_dim = None

    if pooled_text_embedder == "embeddinggemma-300m":
        pooled_projection_dim = 768
    else:
        pooled_projection_dim = None

    if text_embedder == "gemma2-2b":
        context_dim = 2304
    elif text_embedder == "t5":
        context_dim = 4096
    elif text_embedder == "embeddinggemma-300m":
        context_dim = 784
    elif text_embedder == "gemma3-1b":
        context_dim = 1152
    elif text_embedder == "gemma3-4b":
        context_dim = 2560
    elif text_embedder == "gemma3-270m":
        context_dim = 640
    elif text_embedder == "qwen3-4b":
        context_dim = 2560
    elif text_embedder == "qwen3-0.6b":
        context_dim = 1024
    elif text_embedder == "qwen3-1.7b":
        context_dim = 2048
    else:
        raise ValueError(f"Text embedder {text_embedder} not supported")

    denoiser_config = {
        "patch_size": patch_size,
        "in_channels": vae_channels * (patch_size**2),
        "out_channels": vae_channels * (patch_size**2),
        "num_layers": num_layers,
        "num_single_layers": num_single_layers,
        "attention_head_dim": attention_head_dim,
        "num_attention_heads": num_attention_heads,
        "context_dim": context_dim,
        "pooled_projection_dim": pooled_projection_dim,
        "axes_dims_rope": axes_dims_rope,
        "rope_theta": 10_000,
        "mlp_ratio": mlp_ratio,
        "qkv_bias": True,
        "use_batch_norm": use_batch_norm,
        "use_context_rms_norm": use_context_rms_norm,
        "use_pooled_projection_rms_norm": use_pooled_projection_rms_norm,
        "share_modulation": share_modulation,
        "adaln_zero_init": adaln_zero_init,
        "use_adaln_learnable_embedding": use_adaln_learnable_embedding,
        "base_resolution": base_resolution,
        "use_irepa": use_irepa,
        "projectors_target_layers": repa_projectors_target_layers,
        "projectors_hidden_dim": repa_projectors_hidden_dim,
        "projectors_z_dim": repa_projectors_z_dim,
    }

    denoiser = FluxTransformer(**denoiser_config).to(torch.bfloat16)
    logging.info(denoiser)

    # count the number of parameters
    num_params = sum(p.numel() for p in denoiser.parameters())
    logging.info(f"Number of denoiser parameters: {num_params}")

    conditioners = []
    if conditioner_input_key == "text_embedding":
        text_embedder_config = IdentityEmbedderConfig(
            input_key=conditioner_input_key,
            unconditional_conditioning_value=torch.load(
                "/data/clement/working/clipdrop-diffusion/examples/trainings/qwen_4b_empty_string_embed.pth"
            ),
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = IdentityEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)

    elif text_embedder == "gemma2-2b":
        text_embedder_config = GemmaEmbedderConfig(
            version="google/gemma-2-2b",
            text_embedder_subfolder="",
            tokenizer_subfolder="",
            returns_attention_mask=True,
            unconditional_conditioning_value="",
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_padding=tokenizer_padding,
            layer="hidden",
            layer_idx=-2,
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = GemmaEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    elif text_embedder == "embeddinggemma-300m":
        text_embedder_config = Gemma3EmbeddingEmbedderConfig(
            version="google/embeddinggemma-300m",
            output_value="token_embeddings",
            input_key="text",
            unconditional_conditioning_value="",
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = Gemma3EmbeddingEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    elif text_embedder == "gemma3-1b":
        text_embedder_config = Gemma3EmbedderConfig(
            version="google/gemma-3-1b-pt",
            text_embedder_subfolder="",
            tokenizer_subfolder="",
            returns_attention_mask=True,
            unconditional_conditioning_value="",
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_padding=tokenizer_padding,
            layer="hidden",
            layer_idx=-2,
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = Gemma3Embedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    elif text_embedder == "gemma3-4b":
        text_embedder_config = Gemma3EmbedderConfig(
            version="google/gemma-3-4b-pt",
            text_embedder_subfolder="",
            tokenizer_subfolder="",
            returns_attention_mask=True,
            unconditional_conditioning_value="",
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_padding=tokenizer_padding,
            layer="hidden",
            layer_idx=-2,
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = Gemma3Embedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    elif text_embedder == "gemma3-270m":
        text_embedder_config = GemmaEmbedderConfig(
            version="google/gemma-3-270m",
            text_embedder_subfolder="",
            tokenizer_subfolder="",
            returns_attention_mask=True,
            unconditional_conditioning_value="",
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_padding=tokenizer_padding,
            layer="hidden",
            layer_idx=-2,
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = GemmaEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    elif text_embedder == "qwen3-4b":
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
        text_embedder = QwenEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    elif text_embedder == "qwen3-0.6b":
        text_embedder_config = QwenEmbedderConfig(
            version="Qwen/Qwen3-0.6B",
            text_embedder_subfolder="",
            tokenizer_subfolder="",
            returns_attention_mask=True,
            unconditional_conditioning_value="",
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_padding=tokenizer_padding,
            layer_idx=-2,
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = QwenEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    elif text_embedder == "qwen3-1.7b":
        text_embedder_config = QwenEmbedderConfig(
            version="Qwen/Qwen3-1.7B",
            text_embedder_subfolder="",
            tokenizer_subfolder="",
            returns_attention_mask=True,
            unconditional_conditioning_value="",
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_padding=tokenizer_padding,
            layer_idx=-2,
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = QwenEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)
    else:
        text_embedder_config = T5TextEmbedderConfig(
            version="/data/common/checkpoints/FLUX.1-schnell",
            text_embedder_subfolder="text_encoder_2",
            tokenizer_subfolder="tokenizer_2",
            tokenizer_max_length=tokenizer_max_length,
            returns_attention_mask=True,
            use_mask=False,
            unconditional_conditioning_value="",
            unconditional_conditioning_rate=0.1,
            tokenizer_padding=tokenizer_padding,
        )
        text_embedder = T5TextEmbedder(text_embedder_config).to(torch.bfloat16)
        conditioners.append(text_embedder)

    if pooled_text_embedder == "embeddinggemma-300m":
        pooled_text_embedder_config = Gemma3EmbeddingEmbedderConfig(
            version="google/embeddinggemma-300m",
            output_value="sentence_embedding",
            input_key="text",
            unconditional_conditioning_value="",
            unconditional_conditioning_rate=0.1,
        )
        pooled_text_embedder = Gemma3EmbeddingEmbedder(pooled_text_embedder_config).to(
            torch.bfloat16
        )
        conditioners.append(pooled_text_embedder)

    # Freeze text encoders
    text_embedder.freeze()

    # Wrap conditioners
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    ).to(torch.bfloat16)

    ## VAE ##
    # Get VAE model
    if vae_name == "FLUX.1-schnell":
        vae_config = AutoencoderKLDiffusersConfig(
            version="/data/common/checkpoints/FLUX.1-schnell",
            subfolder="vae",
            tiling_size=(128, 128),
            input_key="image",
        )
        vae = AutoencoderKLDiffusers(vae_config).to(torch.bfloat16)

    if vae_name == "FLUX.2-dev":
        vae_config = AutoencoderKLDiffusersConfig(
            version="black-forest-labs/FLUX.2-dev",
            subfolder="vae",
            tiling_size=(128, 128),
            input_key="image",
        )
        vae = AutoencoderKLDiffusers(vae_config).to(torch.bfloat16)

    elif vae_name == "SANA":
        vae_config = AutoencoderDCDiffusersConfig(
            version="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            subfolder="vae",
            tiling_size=(128, 128),
            input_key=vae_input_key,
            dummy_encoder=True if vae_input_key == "latent" else False,
        )
        vae = AutoencoderDCDiffusers(vae_config).to(torch.bfloat16)

    ## Diffusion Model ##
    # Get diffusion model
    config = DiffusionModelConfig(
        input_key=vae_input_key,
        ucg_keys=[conditioner_input_key],
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        logit_mean_schedule=logit_mean_schedule,
        logit_std_schedule=logit_std_schedule,
        timestep_change_steps=timestep_change_steps,
        prediction_type=prediction_type,
        use_ema=use_ema,
        start_ema_after_n_steps=start_ema_after_n_steps,
        ema_update_every_n_steps=ema_update_every_n_steps,
        ema_decay=ema_decay,
        repa_loss_weight=repa_loss_weight if use_irepa else 0.0,
    )

    model = DiffusionModel(
        config,
        denoiser=denoiser,
        vae=vae,
        conditioner=conditioner,
        repa_encoder=repa_encoder,
        repa_encoder_image_processor=repa_encoder_image_processor,
    ).to(torch.bfloat16)
    return model


def get_trainer_and_pipeline(
    model: DiffusionModel,
    config_yaml: dict = None,
    only_denoiser_weight_reload: bool = False,
    resume_from_checkpoint: bool = True,
    start_ckpt: str = None,
    save_ckpt_path: str = None,
    bucket_ckpts: str = "jasper-ai-research",
    run_name: str = None,
    wandb_project: str = "Bloom-Pretraining",
    wandb_tags: List[str] = [],
    log_interval: int = 1000,
    val_check_interval: int = None,
    limit_val_batches: int = 1,
    learning_rate: float = 5e-5,
    learning_rate_scheduler_name: Optional[str] = None,
    learning_rate_scheduler_kwargs: Optional[dict] = {},
    optimizer_name: str = "AdamW",
    optimizer_kwargs: Optional[dict] = {},
    num_steps: int = [20],
    guidance_scale: List[float] = [0.0, 2.5, 5.5, 10, 15],
    shift_value: Optional[List[float]] = None,
    gradient_clip_val: Optional[float] = 1.0,
    save_interval: int = 1000,
    save_interval_for_eval: int = 50000,
    max_steps: int = 400000,
    grad_accumulation: int = 1,
    strategy: str = "ddp",
    validation_prompts: Optional[List[str]] = None,
):
    if only_denoiser_weight_reload and start_ckpt:
        if os.path.exists(start_ckpt):
            state_dict = torch.load(start_ckpt, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
            model.denoiser.load_state_dict(
                {
                    k.replace("model.denoiser.", ""): v
                    for k, v in state_dict.items()
                    if "denoiser" in k and k.startswith("model.denoiser.")
                }
            )
            logging.info(f"Only denoiser weight reload from checkpoint: {start_ckpt}")
            # os.system(f"rm -rf tmp")
        else:
            logging.info(f"Checkpoint not found: {start_ckpt}")
        start_ckpt = None

    ##################### TRAIN #####################
    # Training Config
    training_config = TrainingConfig(
        learning_rates=[learning_rate],
        log_keys=[
            "image",
            "text",
            "caption_name",
            "__url__",
        ],
        trainable_params=[["denoiser."]],
        optimizers_name=[optimizer_name],
        optimizers_kwargs=[optimizer_kwargs],
        lr_schedulers_name=[learning_rate_scheduler_name],
        lr_schedulers_kwargs=[learning_rate_scheduler_kwargs],
        log_samples_model_kwargs={
            "max_samples": 8,
            "num_steps": num_steps,
            "input_shape": None,
            "guidance_scale": guidance_scale,
            "do_guidance": True,
            "cfg_normalization": False,
            "shift_value": shift_value,
            # "conditioner_inputs": (
            #     {
            #         "text_embedding": torch.load(
            #             "/data/clement/working/nano/nano-t2i/examples/trainings/qwen_4b_validation.pth"
            #         ),
            #         "text": (
            #             validation_prompts if validation_prompts is not None else None
            #         ),
            #     }
            # ),
            # "unconditional_conditioner_inputs": (
            #     {
            #         "text": (
            #             unconditional_conditioner_inputs
            #             if unconditional_conditioner_inputs is not None
            #             else None
            #         )
            #     }
            # ),
        },
    )

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    if resume_from_checkpoint and not only_denoiser_weight_reload:
        if os.path.exists(f"{save_ckpt_path}/last.ckpt"):
            start_ckpt = f"{save_ckpt_path}/last.ckpt"
            logging.info(f"Resuming from checkpoint: {start_ckpt}")
        else:
            logging.info(f"Checkpoint not found: {f'{save_ckpt_path}/last.ckpt'}")
            start_ckpt = None
        #         # getting the version of the wandb run
        #         try:
        #             wandb_run_name = torch.load(
        #                 start_ckpt, weights_only=False, map_location="cpu"
        #             )["hyper_parameters"]["wandb_run_name"]
        #             logging.info(f"Resuming from wandb run name: {wandb_run_name}")
        #         except Exception as e:
        #             logging.info(f"Error getting wandb run name: {e}")
        #             wandb_run_name = run_name + f"-{uuid.uuid4()}"
        #             pipeline.save_hyperparameters(
        #                 {
        #                     "wandb_run_name": wandb_run_name,
        #                 }
        #             )
    else:
        #         logging.info(f"Checkpoint not found: {f'tmp/{save_ckpt_path}/last.ckpt'}")
        #         wandb_run_name = run_name + f"-{uuid.uuid4()}"
        #         pipeline.save_hyperparameters(
        #             {
        #                 "wandb_run_name": wandb_run_name,
        #             }
        #         )
        start_ckpt = None

    # else:
    #     start_ckpt = None
    #     wandb_run_name = run_name + f"-{uuid.uuid4()}"
    #     pipeline.save_hyperparameters(
    #         {
    #             "wandb_run_name": wandb_run_name,
    #         }
    #     )

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            # "model": config_yaml.get("model", None),
            # "datasets": config_yaml.get("datasets", None),
            # "training": config_yaml.get("training", None),
            # "logging": config_yaml.get("logging", None),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
        }
    )

    # if rank 0, log to wandb
    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"-{run_name}"
    )
    dir_path = f"{save_ckpt_path}/logs"
    if os.environ["SLURM_PROCID"] == "0":
        os.makedirs(dir_path, exist_ok=True)
    run_name = training_signature

    # Ignore parameters unused during training
    ignore_states = []
    for name, param in pipeline.model.named_parameters():
        ignore = True
        for regex in ["denoiser."]:
            pattern = re.compile(regex)
            if re.match(pattern, name):
                ignore = False
        if ignore:
            ignore_states.append(param)
    if strategy == "fsdp":
        # # FSDP Strategy
        trainer_strategy = FSDPStrategy(
            auto_wrap_policy=ModuleWrapPolicy(
                [
                    FluxTransformer,
                    DoubleStreamBlock,
                    SingleStreamBlock,
                    Modulation,
                    # SelfAttention,
                    # QKNorm,
                    torch.nn.Linear,
                ]
            ),
            activation_checkpointing_policy=ModuleWrapPolicy(
                [
                    DoubleStreamBlock,
                    SingleStreamBlock,
                    # Modulation,
                    # SelfAttention,
                ]
            ),
            sharding_strategy="SHARD_GRAD_OP",
            ignored_states=ignore_states,
        )
    else:
        trainer_strategy = strategy

    logging.info("Setting wandb logger")
    logging.info(f"Run name: {run_name}")
    logging.info(f"Wandb project: {wandb_project}")
    logging.info(f"Wandb tags: {wandb_tags}")
    logging.info(f"Save ckpt path: {save_ckpt_path}")
    logging.info(f"Save interval: {save_interval}")
    logging.info(f"Save interval for eval: {save_interval_for_eval}")
    logging.info(f"Max steps: {max_steps}")
    logging.info(f"Gradient clip val: {gradient_clip_val}")
    logging.info(f"Log interval: {log_interval}")
    logging.info(f"Val check interval: {val_check_interval}")
    logging.info(f"Limit val batches: {limit_val_batches}")
    logging.info(f"Precision: bf16")
    logging.info(f"Check val every n epoch: None")
    logging.info(f"Max steps: {max_steps}")
    logging.info(f"Gradient clip val: {gradient_clip_val}")
    logging.info(f"Log interval: {log_interval}")
    trainer = Trainer(
        accelerator="gpu",
        devices=int(os.environ["SLURM_NPROCS"]) // int(os.environ["SLURM_NNODES"]),
        num_nodes=int(os.environ["SLURM_NNODES"]),
        strategy=trainer_strategy,
        default_root_dir="logs",
        max_epochs=100000,
        gradient_clip_val=gradient_clip_val,
        logger=loggers.WandbLogger(
            project=wandb_project,
            offline=False,
            name=run_name,
            version=run_name + f"-{os.environ['SLURM_JOB_ID']}",
            save_dir=save_ckpt_path,
            tags=wandb_tags,
        ),
        callbacks=(
            [
                WandbSampleLogger(
                    log_batch_freq=log_interval,
                    val_log_batch_freq=(val_check_interval),
                ),
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    dirpath=save_ckpt_path,
                    verbose=True,
                    every_n_train_steps=save_interval,
                    save_last=True,
                    save_top_k=0,
                    enable_version_counter=False,
                ),
                ModelCheckpoint(
                    dirpath=save_ckpt_path,
                    verbose=True,
                    every_n_train_steps=save_interval_for_eval,
                    save_last=False,
                    save_top_k=-1,
                    enable_version_counter=False,
                ),
            ]
            if log_interval is not None
            else [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    dirpath=save_ckpt_path,
                    verbose=True,
                    every_n_train_steps=save_interval,
                    save_last=True,
                    save_top_k=0,
                    enable_version_counter=False,
                ),
                ModelCheckpoint(
                    dirpath=save_ckpt_path,
                    verbose=True,
                    every_n_train_steps=save_interval_for_eval,
                    save_last=False,
                    save_top_k=-1,
                    enable_version_counter=False,
                ),
            ]
        ),
        num_sanity_val_steps=1 if val_check_interval is not None else 0,
        log_every_n_steps=10000000000,
        limit_val_batches=limit_val_batches if val_check_interval is not None else 0,
        val_check_interval=(val_check_interval),
        precision="bf16",
        check_val_every_n_epoch=None if val_check_interval is not None else 100000000,
        max_steps=max_steps,
        accumulate_grad_batches=grad_accumulation,
        # limit_val_batches=20,
        # val_check_interval=log_interval,
    )

    return trainer, pipeline, start_ckpt


def get_model_from_config(config: dict):
    return get_model(**config)


def get_trainer_and_pipeline_from_config(config: dict, model: DiffusionModel):
    trainer_config = config.get("training", None)
    logging_config = config.get("logging", None)
    if trainer_config:
        return get_trainer_and_pipeline(
            model=model,
            **trainer_config,
            **logging_config,
            config_yaml=config,
        )
    else:
        return None


##################### MODEL #####################
def main(
    config: dict = None,
):

    for phase in config.get("phases", []):
        logging.info("-" * 100)
        logging.info(f"Running phase: {phase.get('name')}")
        logging.info("-" * 100)
        model = get_model_from_config(phase.get("model", None))
        data_module = get_data_module_from_config(phase.get("datasets", None))
        trainer, pipeline, start_ckpt = get_trainer_and_pipeline_from_config(
            phase, model
        )

        trainer.fit(
            pipeline,
            data_module,
            ckpt_path=start_ckpt,
            weights_only=False,
        )
        trainer.logger.experiment.finish()
        del trainer, pipeline, start_ckpt
        logging.info("-" * 100)
        logging.info(f"Finished phase: {phase.get('name')}")
        logging.info("-" * 100)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    main(config)


if __name__ == "__main__":
    fire.Fire(main_from_config)

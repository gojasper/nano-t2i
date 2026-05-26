import datetime
import logging
import os
import random
import re
from typing import List, Optional, Tuple

import fire
import PIL
import torch
import yaml
from huggingface_hub import HfFileSystem
from mappers import generic_mappers
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
    IdentityEmbedder,
    IdentityEmbedderConfig,
    QwenEmbedder,
    QwenEmbedderConfig,
)
from nano_t2i.models.transformers.flux_utils import (
    DoubleStreamBlock,
    Modulation,
    SingleStreamBlock,
)
from nano_t2i.models.transformers.tranformers import FluxTransformer
from nano_t2i.models.vae import AutoencoderDCDiffusers, AutoencoderDCDiffusersConfig
from nano_t2i.trainer import TrainingConfig, TrainingPipeline
from nano_t2i.trainer.loggers import WandbSampleLogger
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

PIL.Image.MAX_IMAGE_PIXELS = 933120000
fs = HfFileSystem()

MAPPER_MAP = {
    "text_to_image": generic_mappers,
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

    dataset_sampling_probabilities = []

    for mapper in mappers:
        mapper_name = mapper.get("name")
        dataset_sampling_probabilities.append(
            mapper.get("probability", 1.0 / len(mappers))
        )

        if mapper_name not in MAPPER_MAP:
            raise ValueError(f"Mapper name {mapper_name} not found in MAPPER_MAP")

        shards = mapper.get("shards", [])
        bucket = mapper.get("bucket")
        if bucket is None:
            raise ValueError(
                "A `bucket` (Hugging Face) must be specified for each mapper in the training config."
            )
        prefix = mapper.get("prefix", "")
        if isinstance(shards, str):
            shards = [shards]

        shards_path_or_urls_unbraced = []
        for shard in shards:

            all_tar_files = [
                f"pipe:curl -s -L https://huggingface.co/datasets/{bucket}/resolve/main/{p.removeprefix(f'datasets/{bucket}/')}"
                for p in fs.glob(f"datasets/{bucket}/{shard}/**/*.tar")
            ]
            # all_tar_files = list(braceexpand.braceexpand(f"{prefix}{bucket}/{shard} -"))
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
        mappers = MAPPER_MAP[mapper_name](
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
    unconditional_conditioning_embed_path: Optional[str] = None,
):

    if text_embedder == "qwen3-4b":
        context_dim = 2560
    else:
        raise ValueError(f"Text embedder {text_embedder} not supported")

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
    logging.info(denoiser)

    # count the number of parameters
    num_params = sum(p.numel() for p in denoiser.parameters())
    logging.info(f"Number of denoiser parameters: {num_params}")

    conditioners = []
    if conditioner_input_key == "text_embedding":
        if unconditional_conditioning_embed_path is None:
            raise ValueError(
                "`unconditional_conditioning_embed_path` must be set when "
                "`conditioner_input_key == 'text_embedding'`. It should point to a "
                "pre-computed tensor of the empty-string text embedding (a .pth file)."
            )
        text_embedder_config = IdentityEmbedderConfig(
            input_key=conditioner_input_key,
            unconditional_conditioning_value=torch.load(
                unconditional_conditioning_embed_path
            ),
            unconditional_conditioning_rate=0.1,
        )
        text_embedder = IdentityEmbedder(text_embedder_config).to(torch.bfloat16)
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
    else:
        raise ValueError(f"Text embedder {text_embedder} not supported")

    # Freeze text encoders
    text_embedder.freeze()

    # Wrap conditioners
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    ).to(torch.bfloat16)

    ## VAE ##
    # Get VAE model
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
        prediction_type=prediction_type,
    )

    model = DiffusionModel(
        config,
        denoiser=denoiser,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)
    return model


def get_trainer_and_pipeline(
    model: DiffusionModel,
    config_yaml: dict = None,
    only_denoiser_weight_reload: bool = False,
    resume_from_checkpoint: bool = True,
    start_ckpt: str = None,
    save_ckpt_path: str = None,
    run_name: str = None,
    wandb_project: str = "Nano-T2I",
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
            "shift_value": shift_value,
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
    else:
        start_ckpt = None

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
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
                    torch.nn.Linear,
                ]
            ),
            activation_checkpointing_policy=ModuleWrapPolicy(
                [
                    DoubleStreamBlock,
                    SingleStreamBlock,
                ]
            ),
            sharding_strategy="SHARD_GRAD_OP",
            ignored_states=ignore_states,
        )
    else:
        trainer_strategy = strategy

    logging.info("Setting wandb logger")

    callbacks = [
        WandbSampleLogger(
            log_batch_freq=log_interval,
            val_log_batch_freq=(val_check_interval),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if save_interval is not None and save_interval_for_eval != save_interval:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                verbose=True,
                every_n_train_steps=save_interval,
                save_last=True,
                save_top_k=0,
                enable_version_counter=False,
            )
        )
    if save_interval_for_eval is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                verbose=True,
                every_n_train_steps=save_interval_for_eval,
                save_last=(
                    False
                    if save_interval_for_eval != save_interval
                    and save_interval is not None
                    else True
                ),
                save_top_k=-1,
                enable_version_counter=False,
            )
        )

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
        callbacks=callbacks,
        num_sanity_val_steps=1 if val_check_interval is not None else 0,
        log_every_n_steps=10000000000,
        limit_val_batches=limit_val_batches if val_check_interval is not None else 0,
        val_check_interval=(val_check_interval),
        precision="bf16",
        check_val_every_n_epoch=None if val_check_interval is not None else 100000000,
        max_steps=max_steps,
        accumulate_grad_batches=grad_accumulation,
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

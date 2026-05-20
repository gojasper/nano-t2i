# Adapted from https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from .flux_utils import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    Modulation,
    RMSNorm,
    SingleStreamBlock,
)


@dataclass
class FluxTransformerOutput:
    sample: torch.Tensor
    projected_hs: Optional[Dict[str, torch.Tensor]] = None
    projected_context_hs: Optional[Dict[str, torch.Tensor]] = None


class FluxTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        context_dim: int = 4096,
        pooled_projection_dim: Optional[int] = None,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        rope_theta: int = 10000,
        time_embed_dim: int = 256,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_batch_norm: bool = False,
        use_context_rms_norm: bool = False,
        use_pooled_projection_rms_norm: bool = False,
        share_modulation: bool = False,
        adaln_zero_init: bool = False,
        use_adaln_learnable_embedding: bool = False,
        base_resolution: Optional[Tuple[int, int]] = None,
    ):
        nn.Module.__init__(self)
        assert (
            sum(axes_dims_rope) == attention_head_dim
        ), "sum of axes_dims_rope must be attention_head_dim (got {} != {} attention_head_dim)".format(
            sum(axes_dims_rope), attention_head_dim
        )
        if out_channels is None:
            out_channels = in_channels

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.base_resolution = base_resolution
        self.time_embed = Timesteps(
            num_channels=time_embed_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        self.rope_embedder = EmbedND(
            dim=attention_head_dim,
            theta=rope_theta,
            axes_dim=axes_dims_rope,
        )

        self.latent_embedder = nn.Linear(
            in_features=in_channels, out_features=self.inner_dim, bias=True
        )

        self.time_embedder = MLPEmbedder(
            in_dim=time_embed_dim,
            hidden_dim=self.inner_dim,
        )
        if pooled_projection_dim is not None:
            self.pooled_projection_embedder = MLPEmbedder(
                in_dim=pooled_projection_dim,
                hidden_dim=self.inner_dim,
            )
            if use_pooled_projection_rms_norm:
                logging.info("Using pooled projection RMSNorm")
                self.pooled_projection_rms_norm = RMSNorm(pooled_projection_dim)
            else:
                self.pooled_projection_rms_norm = nn.Identity()
        else:
            self.pooled_projection_embedder = None
        self.context_embedder = nn.Linear(
            context_dim,
            self.inner_dim,
        )
        if use_context_rms_norm:
            logging.info("Using context RMSNorm")
            self.context_rms_norm = RMSNorm(context_dim)
        else:
            self.context_rms_norm = nn.Identity()
        if share_modulation:
            # Store shared modulation layers in a ModuleDict to ensure proper FSDP handling
            logging.info("Using shared modulation")
            self.shared_modulations = nn.ModuleDict()
            if num_single_layers > 0:
                self.shared_modulations["single_stream"] = Modulation(
                    self.inner_dim, double=False
                )
            if num_layers > 0:
                self.shared_modulations["double_stream_text"] = Modulation(
                    self.inner_dim, double=True
                )
                self.shared_modulations["double_stream_img"] = Modulation(
                    self.inner_dim, double=True
                )
        else:
            self.shared_modulations = None

        self.double_blocks = nn.ModuleList()
        for _ in range(num_layers):
            dbl_stream_block = DoubleStreamBlock(
                self.inner_dim,
                num_attention_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_adaln_learnable_embedding=use_adaln_learnable_embedding,
                img_modulation_layer=(
                    self.shared_modulations["double_stream_img"]
                    if share_modulation
                    else None
                ),
                txt_modulation_layer=(
                    self.shared_modulations["double_stream_text"]
                    if share_modulation
                    else None
                ),
            )
            self.double_blocks.append(dbl_stream_block)

        self.single_blocks = nn.ModuleList()
        for _ in range(num_single_layers):
            single_stream_block = SingleStreamBlock(
                self.inner_dim,
                num_attention_heads,
                mlp_ratio=mlp_ratio,
                use_adaln_learnable_embedding=use_adaln_learnable_embedding,
                modulation_layer=(
                    self.shared_modulations["single_stream"]
                    if share_modulation
                    else None
                ),
            )
            self.single_blocks.append(single_stream_block)

        self.last_layer = LastLayer(self.inner_dim, 1, self.out_channels)

        if use_batch_norm:
            self.bn = torch.nn.BatchNorm2d(
                in_channels,
                eps=1e-4,
                momentum=0.1,
                affine=False,
                track_running_stats=True,
            )
            # set me
            self.running_mean = torch.zeros_like(self.bn.running_mean)
            self.running_var = torch.ones_like(self.bn.running_var)
        else:
            self.bn = None

        if adaln_zero_init:
            logging.info("Initializing adaln_zero")
            self.adaln_zero_init()

    def adaln_zero_init(self):
        for layers in self.double_blocks:
            layers.adaln_zero_init()
        for layers in self.single_blocks:
            layers.adaln_zero_init()

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor],
    ):
        """
        The forward pass of the model

        Args:

            sample (torch.Tensor): The input sample
            timesteps (Union[torch.Tensor, float, int]): The timesetps. They should be in [0, 1000].
            conditioning (Dict[str, torch.Tensor]): The conditioning data
        """

        assert isinstance(conditioning, dict), "conditionings must be a dictionary"

        bs = sample.shape[0]
        height = sample.shape[2]
        width = sample.shape[3]

        if self.bn is not None:
            sample = self.bn(sample)

        img_ids = self._prepare_latent_image_ids(
            height=height,
            width=width,
            device=sample.device,
            dtype=sample.dtype,
        )

        pooled_cond = conditioning["cond"].get("vector", None)
        crossattn = conditioning["cond"].get("crossattn", None)

        if pooled_cond is not None and self.pooled_projection_rms_norm is not None:
            pooled_cond = self.pooled_projection_rms_norm(pooled_cond)
            pooled_cond = pooled_cond.to(sample)
        if crossattn is not None and self.context_rms_norm is not None:
            crossattn = self.context_rms_norm(crossattn)
            crossattn = crossattn.to(sample)
        text_ids = (
            torch.zeros(crossattn.shape[1], 3)
            .to(device=sample.device, dtype=sample.dtype)
            .repeat(bs, 1, 1)
        )
        sample_channels = sample.shape[1]

        sample, attention_mask = self._pack_latents(
            latents=sample,
            batch_size=bs,
            num_channels_latents=sample.shape[1],
            height=height,
            width=width,
        )

        noisy_seq_len = sample.shape[1]
        img_ids = img_ids.repeat(bs, 1, 1)

        latent_embed = self.latent_embedder(sample)
        time_embed = self.time_embedder(
            self.time_embed(timestep).to(latent_embed.dtype)
        )

        if self.pooled_projection_embedder is not None:
            time_embed = time_embed + self.pooled_projection_embedder(pooled_cond)

        context_embed = self.context_embedder(crossattn)
        ids = torch.cat((text_ids, img_ids), dim=1)

        rope_embed = self.rope_embedder(ids)

        projected_hs = {}
        projected_context_hs = {}
        layer_id = 0
        for block in self.double_blocks:
            latent_embed, context_embed = block(
                img=latent_embed,
                txt=context_embed,
                vec=time_embed,
                pe=rope_embed,
                attention_mask=attention_mask,
            )

        latent_embed = torch.cat((context_embed, latent_embed), dim=1)

        for block in self.single_blocks:
            latent_embed = block(
                x=latent_embed,
                vec=time_embed,
                pe=rope_embed,
                attention_mask=attention_mask,
            )
        latent_embed = latent_embed[:, context_embed.shape[1] :, ...]

        out = self.last_layer(x=latent_embed, vec=time_embed)[:, :noisy_seq_len]
        return FluxTransformerOutput(
            sample=self._unpack_latents(
                latents=out,
                height=height // self.patch_size,
                width=width // self.patch_size,
            )[:, :sample_channels],
            projected_hs=projected_hs,
            projected_context_hs=projected_context_hs,
        )

    def _prepare_latent_image_ids(
        self, height, width, context_id: int = 0, device=None, dtype=None
    ):
        latent_image_ids = torch.zeros(
            height // self.patch_size, width // self.patch_size, 3
        )
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height // self.patch_size)[:, None]
        ) * (
            self.base_resolution[0] // height if self.base_resolution is not None else 1
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width // self.patch_size)[None, :]
        ) * (
            self.base_resolution[1] // width if self.base_resolution is not None else 1
        )

        latent_image_ids[..., 0] = context_id

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids[None, :]
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width,
            latent_image_id_channels,
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size,
            num_channels_latents,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size,
            (height // self.patch_size) * (width // self.patch_size),
            num_channels_latents * self.patch_size * self.patch_size,
        )
        attention_mask = torch.ones(latents.shape[0], latents.shape[1]).to(
            latents.device, torch.bool
        )

        return latents, attention_mask

    def _unpack_latents(self, latents, height, width):
        batch_size, num_patches, channels = latents.shape

        height = height
        width = width

        latents = latents.view(
            batch_size,
            height,
            width,
            channels // (self.patch_size * self.patch_size),
            self.patch_size,
            self.patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(
            batch_size,
            channels // (self.patch_size * self.patch_size),
            height * self.patch_size,
            width * self.patch_size,
        )

        return latents

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..base.base_model import BaseModel
from ..embedders import ConditionerWrapper
from ..transformers import FluxTransformer
from ..vae import AutoencoderDCDiffusers
from .diffusion_model_config import DiffusionModelConfig


class DiffusionModel(BaseModel):
    """This is the Diffusion Model class which defines the model.

    Args:

        config (DiffusionModelConfig):
            Configuration for the model

        denoiser (FluxTransformer):
            Denoiser to use for the diffusion model. Defaults to None

        vae (Union[AutoencoderKLDiffusers, AutoencoderDCDiffusers]):
            VAE to use for the diffusion model. Defaults to None

        conditioner (ConditionerWrapper):
            Conditioner to use for the diffusion model. Defaults to None
    """

    def __init__(
        self,
        config: DiffusionModelConfig,
        denoiser: FluxTransformer = None,
        vae: Union[AutoencoderDCDiffusers] = None,
        conditioner: ConditionerWrapper = None,
    ):
        BaseModel.__init__(self, config)

        self.vae = vae
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.timestep_sampling = config.timestep_sampling
        self.latent_loss_type = config.latent_loss_type
        self.ucg_keys = config.ucg_keys
        self.prediction_type = config.prediction_type
        self.logit_mean = config.logit_mean
        self.logit_std = config.logit_std
        self.prob = config.prob
        self.selected_timesteps = config.selected_timesteps
        logging.info(
            f"Diffusion model initialized with prediction type: {self.prediction_type}"
        )

        self.iter = nn.Buffer(torch.tensor(0))

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts"""
        super().on_fit_start(device=device, *args, **kwargs)
        if self.vae is not None:
            self.vae.on_fit_start(device=device, *args, **kwargs)
        if self.conditioner is not None:
            self.conditioner.on_fit_start(device=device, *args, **kwargs)

        self.iter.cpu()

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def get_schedule(
        self,
        num_steps: int,
        image_seq_len: int,
        shift_base: float = 4096,
        shift: bool = True,
        shift_value: Optional[float] = None,
    ) -> list[float]:
        # extra step for zero
        timesteps = torch.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            if shift_value is not None:
                time_dist_shift = shift_value
            else:
                time_dist_shift = math.sqrt(image_seq_len / shift_base)

            timesteps = (
                time_dist_shift * timesteps / (1 + (time_dist_shift - 1) * timesteps)
            )

        return timesteps.tolist()

    def forward(self, batch: Dict[str, Any], opt_idx: int = 0, *args, **kwargs):
        start_time = time.perf_counter()
        self.iter += 1

        if batch[self.input_key].ndim > 4:
            batch[self.input_key] = batch[self.input_key].squeeze()
        # Get inputs/latents
        if self.vae is not None and self.input_key != "latent":
            z = self._encode_inputs(batch)
        else:
            z = batch[self.input_key]

        logging.debug(f"Time to get VAE embedding {time.perf_counter() - start_time}")

        start_time_conditioning = time.perf_counter()
        conditioning = self._get_conditioning(
            batch,
            *args,
            **kwargs,
        )
        logging.debug(
            f"Time to get conditioning {time.perf_counter() - start_time_conditioning}"
        )

        # Sample noise
        noise = torch.randn_like(z)

        start_timestep_sampling = time.perf_counter()
        # Sample a timestep
        timestep = self._timestep_sampling(
            image_seq_len=z.shape[1] * z.shape[2] * z.shape[3],
            n_samples=z.shape[0],
            device=z.device,
        ).to(z.dtype)
        logging.debug(
            f"Time to sample timestep {time.perf_counter() - start_timestep_sampling}"
        )

        noisy_sample = (
            timestep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * noise
            + (1.0 - timestep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) * z
        )

        start_time_dit = time.perf_counter()
        # Predict noise level using denoiser
        denoiser_output = self.denoiser(
            sample=(noisy_sample),
            timestep=timestep.to(z.dtype) * 1000,
            conditioning=conditioning,
        )
        prediction = denoiser_output.sample
        logging.debug(f"Time to predict noise {time.perf_counter() - start_time_dit}")

        target = noise - z

        # Compute loss
        start_time_latent_loss = time.perf_counter()
        loss = self.latent_loss(prediction, target)
        logging.debug(
            f"Time to compute latent loss {time.perf_counter() - start_time_latent_loss}"
        )
        out = {"loss": loss.mean(), "latent_loss": loss.mean()}

        logging.debug(f"out: {out}")
        logging.debug(f"Forward time: {time.perf_counter() - start_time}")
        return out

    def latent_loss(self, prediction, model_input):
        if self.latent_loss_type == "l2":
            return torch.mean(
                ((prediction - model_input) ** 2).reshape(model_input.shape[0], -1), 1
            )
        elif self.latent_loss_type == "l1":
            return torch.mean(
                torch.abs(prediction - model_input).reshape(model_input.shape[0], -1),
                1,
            )
        else:
            raise NotImplementedError(
                f"Loss type {self.latent_loss_type} not implemented"
            )

    @torch.no_grad()
    def _encode_inputs(self, batch: Dict[str, Any]):
        """
        Encode the inputs using the VAE
        """
        vae_inputs = batch[self.vae.config.input_key]
        return self.vae.encode(vae_inputs, batch_size=vae_inputs.shape[0])

    @torch.no_grad()
    def _get_conditioning(
        self,
        batch: Dict[str, Any],
        ucg_keys: List[str] = None,
        set_ucg_rate_zero=False,
        *args,
        **kwargs,
    ):
        """
        Get the conditionings
        """
        if self.conditioner is not None:
            return self.conditioner(
                batch,
                ucg_keys=ucg_keys,
                set_ucg_rate_zero=set_ucg_rate_zero,
                vae=self.vae,
                *args,
                **kwargs,
            )
        else:
            return None

    def _timestep_sampling(self, image_seq_len, n_samples=1, device="cpu"):
        timesteps = self.get_schedule(
            num_steps=1000,
            image_seq_len=image_seq_len,
        )

        if self.timestep_sampling == "uniform":
            idx = torch.randint(
                0,
                1000,
                (n_samples,),
                device="cpu",
            )
            return torch.Tensor(timesteps)[idx].to(device=device)

        elif self.timestep_sampling == "log_normal":
            timesteps.reverse()
            u = torch.normal(
                mean=self.logit_mean,
                std=self.logit_std,
                size=(n_samples,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * 1000).long()
            return torch.Tensor(timesteps)[indices].to(device=device)

        elif self.timestep_sampling == "custom_timesteps":
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)

            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_steps: int = 20,
        guidance_scale: float = 1.0,
        conditioner_inputs: Optional[Dict[str, Any]] = None,
        uncond_conditioner_inputs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        do_guidance: bool = True,
        shift_value: Optional[float] = None,
    ):
        """
        Sample from the model

        Args:

            z (torch.Tensor): Noisy latent vector
            num_steps (int): Number of steps to sample. Default: 20
            guidance_scale (float): Guidance scale for classiffier-free guidance. If 1, no guidance. Default: 1.0
            conditioner_inputs (Optional(Dict[str, Any])): inputs to the conditioners. Default: None
            uncond_conditioner_inputs (Optional(Dict[str, Any])): inputs to the conditioners for unconditional conditioning. Default: None
            max_samples (Optional[int]): Maximum number of samples to return. Default: None
            do_guidance (bool): Whether to perform guidance. Note, similar `to guidance_scale` <= 1.0. Default: True
        """
        sample = z
        timesteps = self.get_schedule(
            num_steps=num_steps,
            image_seq_len=z.shape[1] * z.shape[2] * z.shape[3],
            shift=shift_value is not None,
            shift_value=shift_value,
        )

        print(f"timesteps sampling: {timesteps}, shift_value: {shift_value}")

        # Get conditioning
        conditioning = self._get_conditioning(
            conditioner_inputs, set_ucg_rate_zero=True
        )

        if guidance_scale <= 1.0 or not do_guidance:
            unconditional_conditioning = None

        elif uncond_conditioner_inputs is not None:
            unconditional_conditioning = self._get_conditioning(
                uncond_conditioner_inputs,
                set_ucg_rate_zero=True,
            )

        else:
            # Get unconditional conditioning
            unconditional_conditioning = self._get_conditioning(
                conditioner_inputs,
                ucg_keys=self.ucg_keys,
            )

        # If max_samples parameter is provided, limit the number of samples
        if max_samples is not None:
            sample = sample[:max_samples]

        if conditioning:
            conditioning["cond"] = {
                k: v[:max_samples] for k, v in conditioning["cond"].items()
            }
        if unconditional_conditioning:
            unconditional_conditioning["cond"] = {
                k: v[:max_samples]
                for k, v in unconditional_conditioning["cond"].items()
            }

        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            cond_pred = self.denoiser(
                sample=sample,
                timestep=torch.Tensor([t_curr * 1000])
                .to(z.device, dtype=z.dtype)
                .repeat(sample.shape[0]),
                conditioning=conditioning,
            ).sample
            if unconditional_conditioning:
                uncond_pred = self.denoiser(
                    sample=sample,
                    timestep=torch.Tensor([t_curr * 1000])
                    .to(z.device, dtype=z.dtype)
                    .repeat(sample.shape[0]),
                    conditioning=unconditional_conditioning,
                ).sample

                pred = guidance_scale * cond_pred + (1 - guidance_scale) * uncond_pred
                sample = sample + (t_prev - t_curr) * pred
            else:
                sample = sample + (t_prev - t_curr) * cond_pred

        if self.vae is not None:
            decoded_sample = self.vae.decode(sample)

        else:
            decoded_sample = sample

        return decoded_sample

    def log_samples(
        self,
        batch: Dict[str, Any],
        input_shape: Optional[Tuple[int, int, int]] = None,
        guidance_scale: Union[float, List[float]] = 1.0,
        max_samples: int = 1,
        num_steps: Union[int, List[int]] = 20,
        conditioner_inputs: Optional[Dict] = None,
        unconditional_conditioner_inputs: Optional[Dict] = None,
        do_guidance: bool = True,
        shift_value: Optional[Union[float, List[float]]] = None,
    ):
        if isinstance(num_steps, int):
            num_steps = [num_steps]

        if isinstance(guidance_scale, float):
            guidance_scale = [guidance_scale]

        if isinstance(shift_value, float) or shift_value is None:
            shift_value = [shift_value]

        logs = {}

        N = (
            min(max_samples, len(batch[self.input_key]))
            if max_samples is not None
            else len(batch[self.input_key])
        )

        print(f"N: {N}")
        print(f"batch[self.input_key]: {batch[self.input_key].shape}")

        if conditioner_inputs is not None:
            max_conditioning_samples = min(
                [len(conditioner_inputs[key]) for key in conditioner_inputs]
            )
            conditioner_inputs_ = {
                k: v.to(self.device)
                for k, v in conditioner_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            conditioner_inputs.update(conditioner_inputs_)
            batch.update(conditioner_inputs)
            N = min(N, max_conditioning_samples)

        if unconditional_conditioner_inputs is not None:
            max_conditioning_samples = min(
                [
                    len(unconditional_conditioner_inputs[key])
                    for key in unconditional_conditioner_inputs
                ]
            )
            unconditional_conditioner_inputs_ = {
                k: v.to(self.device)
                for k, v in unconditional_conditioner_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            unconditional_conditioner_inputs.update(unconditional_conditioner_inputs_)
            N = min(N, max_conditioning_samples)

        batch = {k: v[:N] for k, v in batch.items()}

        # infer input shape based on VAE configuration if not passed
        if input_shape is None:
            if self.vae is not None:
                # get input pixel size of the vae
                if self.vae.config.input_key != "latent":
                    input_shape = batch[self.vae.config.input_key].shape[2:]
                    # rescale to latent size
                    input_shape = (
                        self.vae.latent_channels,
                        input_shape[0] // self.vae.downsampling_factor,
                        input_shape[1] // self.vae.downsampling_factor,
                    )
                else:
                    input_shape = batch[self.vae.config.input_key].shape[-3:]
                    print(f"input_shape: {input_shape}")
            else:
                raise ValueError(
                    "input_shape must be passed when no VAE is used in the model"
                )

        for num_step in num_steps:
            for guidance in guidance_scale:
                for shift in shift_value:

                    # Log samples
                    z = torch.randn(N, *input_shape).to(self.device, dtype=self.dtype)

                    print(f"z: {z.shape}")

                    logging.debug(
                        f"Sampling {N} samples: steps={num_step}, guidance_scale={guidance}, shift_value={shift_value}"
                    )

                    with torch.autocast(dtype=self.dtype, device_type="cuda"):
                        if shift is None:
                            shift_value_str = "auto"
                        else:
                            shift_value_str = shift

                        logs[
                            f"samples_{num_step}_steps_{guidance}_cfg_{shift_value_str}_shift"
                        ] = self.sample(
                            z,
                            num_steps=num_step,
                            uncond_conditioner_inputs=unconditional_conditioner_inputs,
                            conditioner_inputs=batch,
                            guidance_scale=guidance,
                            max_samples=N,
                            do_guidance=do_guidance,
                            shift_value=shift,
                        )

        return logs

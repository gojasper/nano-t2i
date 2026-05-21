from typing import List, Literal, Optional, Union

from pydantic.dataclasses import dataclass

from ..base import ModelConfig


@dataclass
class DiffusionModelConfig(ModelConfig):
    """This is the Config for Diffusion Model class which defines all the useful parameters to be used in the model.

    Args:

        latent_loss_type (str):
            Loss type to use. Defaults to "l2". Choices are "l2", "l1"

        timestep_sampling (str):
            Timestep sampling to use. Defaults to "uniform". Choices are ["uniform", "log_normal", "custom_timesteps"]

        input_key (str):
            Key for the input. Defaults to "image"

        ucg_keys (Optional[List[str]]):
            List of keys for which we enforce zero_conditioning during Classifier-free guidance. Defaults to None

        prediction_type (str):
            Type of prediction to use. Defaults to "epsilon". Choices are "epsilon", "v_prediction",
            "flow_matching", "flow_matching_sd3", "sample"

        logit_mean (Optional[float]):
            Mean of the logit for the log normal distribution. Defaults to 0.0

        logit_std (Optional[float]):
            Standard deviation of the logit for the log normal distribution. Defaults to 1.0

        selected_timesteps (Optional[List[float]]):
            List of selected timesteps to be sampled from if using `custom_timesteps` timestep sampling. Defaults to None

        prob (Optional[List[float]]):
            List of probabilities for the selected timesteps if using `custom_timesteps` timestep sampling. Defaults to None
    """

    latent_loss_type: Literal["l2", "l1"] = "l2"
    timestep_sampling: Literal["uniform", "log_normal", "custom_timesteps"] = "uniform"
    input_key: str = "image"
    ucg_keys: Optional[List[str]] = None
    prediction_type: Literal["epsilon", "v_prediction", "flow_matching", "sample"] = (
        "flow_matching"
    )
    logit_mean: Optional[Union[float, List[float]]] = 0.0
    logit_std: Optional[Union[float, List[float]]] = 1.0
    selected_timesteps: Optional[List[float]] = None
    prob: Optional[List[float]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.timestep_sampling == "custom_timesteps":
            assert isinstance(self.selected_timesteps, list) and isinstance(
                self.prob, list
            ), "timesteps and prob should be list for custom_timesteps timestep sampling"
            assert len(self.selected_timesteps) == len(
                self.prob
            ), "timesteps and prob should be of same length for custom_timesteps timestep sampling"
            assert (
                sum(self.prob) == 1
            ), "prob should sum to 1 for custom_timesteps timestep sampling"

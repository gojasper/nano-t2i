import logging
from typing import Callable, List, Optional, Union

import webdataset as wds
from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class DataModuleConfig(BaseConfig):
    """
    Configuration for the DataModule

    Args:

        shards_path_or_urls (Union[str, List[str]]): The path or url to the shards. Defaults to None.
        per_worker_batch_size (int): The batch size for the dataset. Defaults to 16.
        num_workers (int): The number of workers to use. Defaults to 1.
        shuffle_before_split_by_node_buffer_size (Optional[int]): The buffer size for the shuffle before split by node. Defaults to 100.
        shuffle_before_split_by_workers_buffer_size (Optional[int]): The buffer size for the shuffle before split by workers. Defaults to 100.
        shuffle_before_filter_mappers_buffer_size (Optional[int]): The buffer size for the shuffle before filter mappers. Defaults to 1000.
        shuffle_after_filter_mappers_buffer_size (Optional[int]): The buffer size for the shuffle after filter mappers. Defaults to 1000.
        decoder (str): The decoder to use. Defaults to "pil".
        handler (Callable): A callable to handle the warnings. Defaults to wds.warn_and_continue.
    """

    shards_path_or_urls: Union[str, List[str]] = None
    per_worker_batch_size: int = 16
    num_workers: int = 1
    shuffle_before_split_by_node_buffer_size: Optional[int] = 100
    shuffle_before_split_by_workers_buffer_size: Optional[int] = 100
    shuffle_before_filter_mappers_buffer_size: Optional[int] = 1000
    shuffle_after_filter_mappers_buffer_size: Optional[int] = 1000
    decoder: Optional[str] = "pil"
    handler: Callable = wds.warn_and_continue

    def __post_init__(self):
        super().__post_init__()

        if self.shards_path_or_urls is not None:
            logging.debug(
                f"DataModule received {len(self.shards_path_or_urls)} shards: {self.shards_path_or_urls[:10]}"
            )


@dataclass
class MultiDataModuleConfig(BaseConfig):
    """
    Configuration for the MultiDataModule

    Args:
        configs (List[DataModuleConfig]): The configurations for the datasets. Note that per_worker_batch_size must be set to 1 for each config. Batching is handled internally.
        dataset_sampling_probabilities (List[float]): The probabilities for the datasets.
        per_worker_batch_size (int): The batch size for the dataset.
        per_dataset_pixel_budgets (List[int]): The pixel budgets for the each of the datasets.
        per_dataset_pixel_budgets_probabilities (List[float]): The probabilities for the pixel budgets for each of the datasets.
    """

    configs: List[DataModuleConfig]
    dataset_sampling_probabilities: Optional[List[float]] = None
    per_worker_batch_size: int = 16
    per_dataset_pixel_budgets: Optional[Union[List[int], List[List[int]]]] = None
    per_dataset_pixel_budgets_probabilities: Optional[
        Union[List[float], List[List[float]]]
    ] = None

    def __post_init__(self):
        super().__post_init__()
        for config in self.configs:
            assert (
                config.per_worker_batch_size == 1
            ), "per_worker_batch_size must be set to 1 when using MultiDataModule. Batching is handled internally."

        if self.dataset_sampling_probabilities is not None:
            assert len(self.dataset_sampling_probabilities) == len(
                self.configs
            ), "dataset_sampling_probabilities must have the same length as configs"

        if (
            self.per_dataset_pixel_budgets_probabilities is not None
            or self.per_dataset_pixel_budgets is not None
        ):
            assert (
                self.per_dataset_pixel_budgets_probabilities is not None
                and self.per_dataset_pixel_budgets is not None
            ), "probabilities and pixel_budgets must be provided"

            assert len(self.per_dataset_pixel_budgets) == len(
                self.configs
            ), "per_dataset_pixel_budgets must have the same length as configs"
            assert len(self.per_dataset_pixel_budgets_probabilities) == len(
                self.configs
            ), "per_dataset_pixel_budgets_probabilities must have the same length as configs"

            for (
                per_dataset_pixel_budget,
                per_dataset_pixel_budgets_probabilities,
            ) in zip(
                self.per_dataset_pixel_budgets,
                self.per_dataset_pixel_budgets_probabilities,
            ):
                assert len(per_dataset_pixel_budget) == len(
                    per_dataset_pixel_budgets_probabilities
                ), "per_dataset_pixel_budget and per_dataset_pixel_budgets_probabilities must have the same length"
                assert (
                    sum(per_dataset_pixel_budgets_probabilities) == 1
                ), "per_dataset_pixel_budgets_probabilities must be sum to 1"

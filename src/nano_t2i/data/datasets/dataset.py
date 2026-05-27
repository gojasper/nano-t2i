from functools import partial
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import webdataset as wds
from webdataset import DataPipeline
from webdataset.cache import FileCache

from ..filters import BaseFilter, FilterWrapper
from ..mappers import BaseMapper, MapperWrapper
from ..mappers_batched import BaseMapperBatched, MapperBatchedWrapper
from .collation_fn import custom_collation_fn
from .datasets_config import DataModuleConfig, MultiDataModuleConfig
from .utils import RandomSampleMultiDatasets


class DataPipeline:
    """
    DataPipeline class for creating a dataloader from a single configuration

    Args:

        config (DataModuleConfig):
            Configuration for the dataset

        filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the dataset. These will be sequentially applied.

        batched_filters_mappers (Union[BaseMapperBatched, MapperBatchedWrapper]):
            List of batched transforms for the dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        config: DataModuleConfig,
        filters_mappers: Optional[
            List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]
        ] = None,
        batched_filters_mappers: Optional[
            Union[BaseMapperBatched, MapperBatchedWrapper]
        ] = None,
    ):
        self.config = config
        self.shards_path_or_urls = config.shards_path_or_urls
        self.filters_mappers = filters_mappers
        self.batched_filters_mappers = batched_filters_mappers or []
        if not isinstance(self.batched_filters_mappers, list):
            self.batched_filters_mappers = [self.batched_filters_mappers]

        if self.filters_mappers is None:
            self.filters_mappers = []

        # set processing pipeline
        if config.decoder is not None:
            self.processing_pipeline = [
                wds.decode(config.decoder, handler=config.handler)
            ]
        else:
            self.processing_pipeline = [wds.decode(handler=config.handler)]
        self.processing_pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=self.filters_mappers,
                handler=config.handler,
            )
        )

    def _add_filters_mappers(
        self,
        filters_mappers: List[
            Union[FilterWrapper, MapperWrapper, MapperBatchedWrapper]
        ],
        handler: Callable = wds.warn_and_continue,
    ) -> List[Union[FilterWrapper, MapperWrapper, MapperBatchedWrapper]]:
        tmp_pipeline = []
        for filter_mapper in filters_mappers:
            if isinstance(filter_mapper, FilterWrapper) or isinstance(
                filter_mapper, BaseFilter
            ):
                tmp_pipeline.append(wds.select(filter_mapper))
            elif isinstance(filter_mapper, MapperWrapper) or isinstance(
                filter_mapper, BaseMapper
            ):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            elif isinstance(filter_mapper, MapperBatchedWrapper) or isinstance(
                filter_mapper, BaseMapperBatched
            ):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            else:
                raise ValueError("Unknown type of filter/mapper")
        return tmp_pipeline

    def setup(self, batching: bool = True):
        pipeline = [wds.SimpleShardList(self.shards_path_or_urls)]

        # shuffle before split by node
        if self.config.shuffle_before_split_by_node_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_split_by_node_buffer_size,
                    handler=self.config.handler,
                )
            )
        # split by node
        pipeline.append(wds.split_by_node)

        # shuffle before split by workers
        if self.config.shuffle_before_split_by_workers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_split_by_workers_buffer_size,
                    handler=self.config.handler,
                )
            )
        # split by worker
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(
                    handler=self.config.handler,
                ),
            ]
        )

        # shuffle before filter mappers
        if self.config.shuffle_before_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_filter_mappers_buffer_size,
                    handler=self.config.handler,
                )
            )

        # apply filters and mappers
        pipeline.extend(self.processing_pipeline)

        # shuffle after filter mappers
        if self.config.shuffle_after_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_after_filter_mappers_buffer_size,
                    handler=self.config.handler,
                ),
            )

        if batching:
            # batching
            pipeline.append(
                wds.batched(
                    self.config.per_worker_batch_size,
                    collation_fn=custom_collation_fn,
                )
            )

        # apply batched transforms
        pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=self.batched_filters_mappers,
                handler=self.config.handler,
            )
        )

        # create the data pipeline
        pipeline = wds.DataPipeline(*pipeline, handler=self.config.handler)

        # set the pipeline
        self.pipeline = pipeline

    def dataloader(self):
        # return the loader
        return wds.WebLoader(
            self.pipeline,
            batch_size=None,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=8,
        )


class MultiDataPipeline(DataPipeline):
    """
    MultiDataPipeline class for creating a dataloader from multiple configurations

    Args:

        config (MultiDataModuleConfig):
            Configuration for the multi data pipeline.

        filters_mappers (List[
            Union[
                List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]],
                BaseMapper,
                BaseFilter,
                FilterWrapper,
                MapperWrapper,
            ]
        ]):
            List of filters and mappers for each of the training datasets. These will be sequentially applied.

        batched_filters_mappers (Union[BaseMapperBatched, MapperBatchedWrapper]):
            Mappers to be applied to gather the samples from each dataset.

        dataset_sampling_probabilities (List[float]):
            List of probabilities for sampling for each training datasets. If None, the datasets will be sampled uniformly.
            Default is None, sampled uniformly.

        per_dataset_pixel_budgets (List[List[int]]):
            List of pixel budgets for each of the datasets.

        per_dataset_pixel_budgets_probabilities (List[List[float]]):
            List of probabilities for the pixel budgets for each of the datasets.
    """

    def __init__(
        self,
        config: MultiDataModuleConfig,
        filters_mappers: Optional[
            List[
                Union[
                    List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]],
                    BaseMapper,
                    BaseFilter,
                    FilterWrapper,
                    MapperWrapper,
                ]
            ]
        ] = None,
        batched_filters_mappers: Optional[
            Union[BaseMapperBatched, MapperBatchedWrapper]
        ] = None,
    ):

        configs = config.configs
        self.config = config

        assert len(configs) > 0, "configs must have at least one configuration"
        self.configs = configs

        # check if filters_mappers is None and length match
        if filters_mappers is not None and len(filters_mappers) != len(configs):
            raise ValueError(
                "filters_mappers must have the same length as configs",
                len(filters_mappers),
                len(configs),
            )

        # set sampling probabilities
        if config.dataset_sampling_probabilities is not None:
            if len(config.dataset_sampling_probabilities) != len(configs):
                raise ValueError(
                    "dataset_probabilities must have the same length as configs"
                )
            self.dataset_sampling_probabilities = config.dataset_sampling_probabilities
        else:
            # set uniform sampling
            self.dataset_sampling_probabilities = [1 / len(configs) for _ in configs]
        self.per_dataset_pixel_budgets = self.config.per_dataset_pixel_budgets
        self.per_dataset_pixel_budgets_probabilities = (
            self.config.per_dataset_pixel_budgets_probabilities
        )

        # transform to list if None
        if filters_mappers is None:
            self.filters_mappers = [None for _ in configs]
        else:
            self.filters_mappers = filters_mappers
        self.batched_filters_mappers = batched_filters_mappers

    def setup(self, batching: bool = True):
        pipelines = []

        # iterate over all configurations
        for i, config in enumerate(self.configs):
            pipeline = DataPipeline(
                config=config,
                filters_mappers=self.filters_mappers[i],
                batched_filters_mappers=None,
            )
            pipeline.setup(batching=batching)
            pipelines.append(pipeline.pipeline)

        # set pipeline
        self.pipeline = RandomSampleMultiDatasets(
            datasets=pipelines,
            batch_size=self.config.per_worker_batch_size,
            probabilities=self.dataset_sampling_probabilities,
            batched_mappers=self.batched_filters_mappers,
            per_dataset_pixel_budgets=self.per_dataset_pixel_budgets,
            per_dataset_pixel_budgets_probabilities=self.per_dataset_pixel_budgets_probabilities,
        )

    def dataloader(self):
        # num workers as the minimum number of workers
        num_workers = min(
            [config.num_workers for config in self.configs if config.num_workers]
        )
        # return the loader
        return wds.WebLoader(
            self.pipeline,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=8,
        )


class DataModule(pl.LightningDataModule):
    """
    Main DataModule class for creating data loaders and training/evaluating models

    Args:

        train_config (DataModuleConfig):
            Configuration for the training dataset

        train_filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the training dataset. These will be sequentially applied.

        train_batched_filters_mappers (Union[BaseMapperBatched, MapperBatchedWrapper]):
            List of batched transforms for the training dataset. These will be sequentially applied.

        eval_config (DataModuleConfig):
            Configuration for the evaluation dataset

        eval_filters_mappers (List[Union[FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the evaluation dataset.These will be sequentially applied.

        eval_batched_filters_mappers (Union[BaseMapperBatched, MapperBatchedWrapper]):
            List of batched transforms for the evaluation dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        train_config: DataModuleConfig,
        train_filters_mappers: Optional[
            List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]
        ] = None,
        train_batched_filters_mappers: Optional[
            Union[BaseMapperBatched, MapperBatchedWrapper]
        ] = None,
        eval_config: Optional[DataModuleConfig] = None,
        eval_filters_mappers: Optional[
            List[Union[FilterWrapper, MapperWrapper]]
        ] = None,
        eval_batched_filters_mappers: Optional[
            Union[BaseMapperBatched, MapperBatchedWrapper]
        ] = None,
    ):
        super().__init__()

        self.train_config = train_config
        self.train_filters_mappers = train_filters_mappers
        self.train_batched_filters_mappers = train_batched_filters_mappers

        self.eval_config = eval_config
        self.eval_filters_mappers = eval_filters_mappers
        self.eval_batched_filters_mappers = eval_batched_filters_mappers

    def setup(self, stage=None):
        """
        Setup the data module and create the webdataset processing pipelines
        """

        # train pipeline
        self.train_pipeline = DataPipeline(
            config=self.train_config,
            filters_mappers=self.train_filters_mappers,
            batched_filters_mappers=self.train_batched_filters_mappers,
        )
        self.train_pipeline.setup()

        # eval pipeline
        if self.eval_config is not None:
            self.eval_pipeline = DataPipeline(
                config=self.eval_config,
                filters_mappers=self.eval_filters_mappers,
                batched_filters_mappers=self.eval_batched_filters_mappers,
            )
            self.eval_pipeline.setup()

    def train_dataloader(self):
        return self.train_pipeline.dataloader()

    def val_dataloader(self):
        return self.eval_pipeline.dataloader()


class MultiDataModule(pl.LightningDataModule):
    """MultiDataModule

    DataModule handling multiple data pipeline with different configurations and sampling probabilities

    Args:
        train_config (MultiDataModuleConfig): Training configuration for the multi data module
        train_filters_mappers (List[ Union[ List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]], BaseMapper, BaseFilter, FilterWrapper, MapperWrapper, ] ], optional): Training filters and mappers
        train_batched_filters_mappers (List[ Union[ List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]], BaseMapper, BaseFilter, FilterWrapper, MapperWrapper, ] ], optional): Training batched filters and mappers
        train_dataset_sampling_probabilities (List[float ], optional): Training dataset sampling probabilities
        eval_configs (List[DataModuleConfig], optional): Evaluation configurations
        eval_filters_mappers (List[ Union[ List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]], BaseMapper, BaseFilter, FilterWrapper, MapperWrapper, ] ], optional): Evaluation filters and mappers
        eval_batched_filters_mappers (List[ Union[ List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]], BaseMapper, BaseFilter, FilterWrapper, MapperWrapper, ] ], optional): Evaluation batched filters and mappers
        eval_dataset_sampling_probabilities (List[float], optional): Evaluation dataset sampling probabilities
    """

    def __init__(
        self,
        train_config: MultiDataModuleConfig,
        train_filters_mappers: List[
            Union[
                List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]],
                BaseMapper,
                BaseFilter,
                FilterWrapper,
                MapperWrapper,
            ]
        ] = None,
        train_batched_filters_mappers: Union[
            BaseMapperBatched, MapperBatchedWrapper
        ] = None,
        eval_config: Optional[MultiDataModuleConfig] = None,
        eval_filters_mappers: Optional[
            List[
                Union[
                    List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]],
                    BaseMapper,
                    BaseFilter,
                    FilterWrapper,
                    MapperWrapper,
                ]
            ]
        ] = None,
        eval_batched_filters_mappers: Optional[
            Union[BaseMapperBatched, MapperBatchedWrapper]
        ] = None,
    ):
        super().__init__()

        # Training
        self.global_train_config = train_config
        self.train_configs = train_config.configs
        self.train_filters_mappers = train_filters_mappers
        self.train_batched_filters_mappers = train_batched_filters_mappers
        self.train_dataset_sampling_probabilities = (
            train_config.dataset_sampling_probabilities
        )
        self.train_per_dataset_pixel_budgets = train_config.per_dataset_pixel_budgets
        self.train_per_dataset_pixel_budgets_probabilities = (
            train_config.per_dataset_pixel_budgets_probabilities
        )
        # no batching in per dataset pipeline if there is no batched filters on. Batching is handled in RandomSampleMultiDatasets
        self.train_batching = train_batched_filters_mappers is not None
        if self.train_dataset_sampling_probabilities is None:
            # set uniform sampling
            self.train_dataset_sampling_probabilities = [
                1 / len(self.train_configs) for _ in self.train_configs
            ]

        # Evaluation
        self.global_eval_config = eval_config

        if eval_config is not None:
            self.eval_configs = eval_config.configs
            self.eval_filters_mappers = eval_filters_mappers
            self.eval_batched_filters_mappers = eval_batched_filters_mappers
            self.eval_dataset_sampling_probabilities = (
                eval_config.dataset_sampling_probabilities
            )
            # no batching in per dataset pipeline if there is no batched filters. Batching is handled RandomSampleMultiDatasets
            self.eval_batching = eval_batched_filters_mappers is not None
            if self.eval_dataset_sampling_probabilities is None:
                # set uniform sampling
                self.eval_dataset_sampling_probabilities = [
                    1 / len(self.eval_configs) for _ in self.eval_configs
                ]
            self.eval_per_dataset_pixel_budgets = eval_config.per_dataset_pixel_budgets
            self.eval_per_dataset_pixel_budgets_probabilities = (
                eval_config.per_dataset_pixel_budgets_probabilities
            )
        else:
            self.eval_configs = None

    def setup(self, stage=None):
        # train pipeline
        self.train_pipeline = MultiDataPipeline(
            config=self.global_train_config,
            filters_mappers=self.train_filters_mappers,
            batched_filters_mappers=self.train_batched_filters_mappers,
        )
        self.train_pipeline.setup(batching=self.train_batching)

        # eval pipeline
        if self.eval_configs is not None:
            self.eval_pipeline = MultiDataPipeline(
                config=self.global_eval_config,
                filters_mappers=self.eval_filters_mappers,
                batched_filters_mappers=self.eval_batched_filters_mappers,
            )
            self.eval_pipeline.setup(batching=self.eval_batching)

    def train_dataloader(self):
        return self.train_pipeline.dataloader()

    def val_dataloader(self):
        return self.eval_pipeline.dataloader()

import logging
from typing import List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset
from webdataset import DataPipeline

from ..mappers_batched import MultiAspectRatioCacherMapper
from .collation_fn import custom_collation_fn


class RandomSampleMultiDatasets(IterableDataset):
    """
    Randomly sample from multiple datasets with given probabilities.

    Args:

        datasets (List[Union[IterableDataset, DataPipeline]]): list of datasets to sample from
        batch_size (int): batch size for the batched mapper. Defaults to 1
        probabilities (List[float]): list of probabilities for each dataset. If None, the datasets will be sampled
        uniformly. Defaults to None
        batched_mappers (Union[MultiAspectRatioCacherMapper, MapperBatchedWrapper]): Mappers to be applied to gather the samples from each dataset.
        pixel_budgets (List[int]): List of pixel budgets for each of the datasets.
        pixel_budgets_probabilities (List[float]): List of probabilities for the pixel budgets for each of the datasets.
    """

    def __init__(
        self,
        datasets: List[Union[IterableDataset, DataPipeline]],
        batch_size: int = 1,
        probabilities: Optional[List[float]] = None,
        batched_mappers: Optional[MultiAspectRatioCacherMapper] = None,
        per_dataset_pixel_budgets: Optional[List[List[int]]] = None,
        per_dataset_pixel_budgets_probabilities: Optional[List[List[float]]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.probabilities = probabilities
        self.n_datasets = len(datasets)
        self.batched_mappers = batched_mappers
        self.max_length = max_length
        if per_dataset_pixel_budgets is None:
            per_dataset_pixel_budgets = [None] * self.n_datasets

        if per_dataset_pixel_budgets_probabilities is None:
            per_dataset_pixel_budgets_probabilities = [None] * self.n_datasets

        self.per_dataset_pixel_budgets = per_dataset_pixel_budgets
        self.per_dataset_pixel_budgets_probabilities = (
            per_dataset_pixel_budgets_probabilities
        )
        logging.debug(f"Pixel budgets: {self.per_dataset_pixel_budgets}")
        logging.debug(
            f"Pixel budgets probabilities: {self.per_dataset_pixel_budgets_probabilities}"
        )
        assert self.n_datasets > 0, "datasets should not be empty"

        if probabilities is not None:
            assert len(datasets) == len(
                probabilities
            ), "datasets and probabilities should have the same length"
            assert all(
                0 <= p <= 1 for p in probabilities
            ), "probabilities should be between 0 and 1"
            assert sum(probabilities) == 1, "probabilities should sum to 1"

        else:
            # set uniform sampling
            probabilities = [1 / self.n_datasets] * self.n_datasets

        self.probabilities = probabilities
        self.iterators = [
            self.dataset_cycle_generator(dataset) for dataset in self.datasets
        ]

    def dataset_cycle_generator(self, dataset):
        while True:
            yield from dataset

    def iterable_batched_mapper(self):
        if self.batched_mappers is None:
            batch = []
            for _ in range(self.batch_size):
                batch.append(next(self.dataset_iterator()))
            yield custom_collation_fn(batch, max_length=self.max_length)
        else:
            iterator = self.dataset_iterator(return_dataset_id=True)
            next_batch, dataset_id = next(iterator)
            b = self.batched_mappers(
                next_batch,
                pixel_budgets=self.per_dataset_pixel_budgets[dataset_id],
                pixel_budgets_probabilities=self.per_dataset_pixel_budgets_probabilities[
                    dataset_id
                ],
            )
            if b is not None:
                yield b
            else:
                yield from self.iterable_batched_mapper()

    def dataset_iterator(self, return_dataset_id: bool = False):
        """
        Randomly select a data stream from the list of datasets with given probabilities and return the next sample from the selected dataset. Id is needed for the batched mapper to know which pixel budget to use for each stream.

        Args:
            return_dataset_id (bool): Whether to return the dataset id. Defaults to False.

        Returns:
        """
        dataset_id = np.random.choice(self.n_datasets, p=self.probabilities)
        logging.debug(f"Sampling from dataset {dataset_id}")
        if return_dataset_id:
            yield next(self.iterators[dataset_id]), dataset_id
        else:
            yield next(self.iterators[dataset_id])

    def __iter__(self):
        while True:
            try:
                yield next(self.iterable_batched_mapper())
            except StopIteration as e:
                return

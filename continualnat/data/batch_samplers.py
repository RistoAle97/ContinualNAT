from typing import Iterator, List, Union

import numpy as np
from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler, Sampler

from continualnat.data.datasets import TranslationDataset


class BatchSamplerCore(Sampler[List[int]]):
    def __init__(
        self,
        datasets: Union[ConcatDataset[TranslationDataset], List[TranslationDataset]],
        bsz: int,
        drop_last: bool = False,
        sampling_strategy: str = "random",
    ) -> None:
        """
        Base class for all the batch samplers.
        :param datasets: a list of TranslationDataset or a ConcatDataset.
        :param bsz: the batch size.
        :param drop_last: whether to drop the last batch if its length does not reach the batch size (default=False).
        :param sampling_strategy: the sampling strategy to apply on each dataset, can be either "random"
            or "sequential" (default="random").
        """
        super().__init__(datasets)
        self.concat_dataset = datasets if isinstance(datasets, ConcatDataset) else ConcatDataset(datasets)
        if sampling_strategy not in ["random", "sequential"]:
            raise ValueError("The sampling strategy must be one of \"random\" and \"sequential\".")

        if sampling_strategy == "random":
            self._iter_samplers = [iter(RandomSampler(dataset)) for dataset in self.concat_dataset.datasets]
        else:
            self._iter_samplers = [iter(SequentialSampler(dataset)) for dataset in self.concat_dataset.datasets]

        self.bsz = bsz
        self.max_len = max([len(dataset) for dataset in self.concat_dataset.datasets])
        self.sampling_strategy = sampling_strategy
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            n_batches = self.max_len // self.bsz * len(self._iter_samplers)
        else:
            n_batches = (self.max_len + self.bsz - 1) // self.bsz * len(self._iter_samplers)

        return n_batches

    def __iter__(self) -> Iterator[List[int]]:
        raise NotImplementedError


class HeterogeneousSampler(BatchSamplerCore):
    def __init__(
        self,
        datasets: Union[ConcatDataset[TranslationDataset], List[TranslationDataset]],
        bsz: int,
        drop_last: bool = False,
        sampling_strategy: str = "random",
        weights: List[int] = None,
    ) -> None:
        """
        Samples indices in a way that there will be different translation directions in the same batch.
        :param datasets: a list of TranslationDataset or a ConcatDataset.
        :param bsz: the batch size.
        :param drop_last: whether to drop the last batch if its length does not reach the batch size (default=False).
        :param sampling_strategy: the sampling strategy to apply on each dataset, can be either "random"
            or "sequential" (default="random").
        :param weights: the sampling probability for each dataset. If None then the datasets will share the same weight
            (1 / number of datasets), otherwise the values must sum up to 1.0 (default=None).
        """
        super().__init__(datasets, bsz, drop_last, sampling_strategy)
        self.weights = weights if weights is not None else [1 / len(self._iter_samplers)] * len(self._iter_samplers)
        if sum(self.weights) != 1.0:
            raise ValueError("The weights must sum to 1.")

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        np_generator = np.random.default_rng()
        # Even though it is not entirely correct, we decided to leave the same number of iterations as the homogeneous
        # sampler.
        for _ in range(len(self) * self.bsz):
            sampler_idx = np_generator.choice(len(self._iter_samplers), p=self.weights)
            try:
                sample = next(self._iter_samplers[sampler_idx])
            except StopIteration:
                next_dataset = self.concat_dataset.datasets[sampler_idx]
                if self.sampling_strategy == "random":
                    sampler = iter(RandomSampler(next_dataset))
                else:
                    sampler = iter(SequentialSampler(next_dataset))

                self._iter_samplers[sampler_idx] = sampler
                sample = next(self._iter_samplers[sampler_idx])

            if sampler_idx == 0:
                start = 0
            else:
                start = self.concat_dataset.cumulative_sizes[sampler_idx - 1]

            batch.append(start + sample)
            if len(batch) == self.bsz:
                yield batch
                batch = []

        if not self.drop_last and len(batch) != 0:
            yield batch


class HomogeneousSampler(BatchSamplerCore):
    def __init__(
        self,
        datasets: Union[ConcatDataset[TranslationDataset], List[TranslationDataset]],
        bsz: int,
        drop_last: bool = False,
        sampling_strategy: str = "random",
    ) -> None:
        """
        Sample indices in a way that there will only a single translation direction in a batch.
        :param datasets: a list of TranslationDataset or a ConcatDataset.
        :param bsz: the batch size.
        :param drop_last: whether to drop the last batch if its length does not reach the batch size (default=False).
        :param sampling_strategy: the sampling strategy to apply on each dataset, can be either "random"
            or "sequential" (default="random").
        """
        super().__init__(datasets, bsz, drop_last, sampling_strategy)
        self._current_sampler_idx = 0

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for _ in range(len(self) * self.bsz):
            try:
                sample = next(self._iter_samplers[self._current_sampler_idx])
            except StopIteration:
                dataset_to_oversample = self.concat_dataset.datasets[self._current_sampler_idx]
                if self.sampling_strategy == "random":
                    # noinspection PyTypeChecker
                    sampler = iter(RandomSampler(dataset_to_oversample))
                else:
                    # noinspection PyTypeChecker
                    sampler = iter(SequentialSampler(dataset_to_oversample))

                self._iter_samplers[self._current_sampler_idx] = sampler
                sample = next(self._iter_samplers[self._current_sampler_idx])

            if self._current_sampler_idx == 0:
                start = 0
            else:
                start = self.concat_dataset.cumulative_sizes[self._current_sampler_idx - 1]

            batch.append(start + sample)
            if len(batch) == self.bsz:
                self._current_sampler_idx = (self._current_sampler_idx + 1) % len(self._iter_samplers)
                yield batch
                batch = []

        if not self.drop_last and len(batch) != 0:
            self._current_sampler_idx = (self._current_sampler_idx + 1) % len(self._iter_samplers)
            yield batch

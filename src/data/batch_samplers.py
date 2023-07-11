import numpy as np
from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler
from typing import List, Union
from src.data.datasets import TranslationDataset


class BatchSamplerCore:

    def __init__(self,
                 datasets: Union[ConcatDataset, List[TranslationDataset]],
                 bsz: int,
                 drop_last: bool = False,
                 sampling_strategy: str = "random") -> None:
        """
        Base class for all the batch samplers.
        :param datasets: a list of TranslationDataset or a ConcatDataset.
        :param bsz: the batch size.
        :param drop_last: whether to drop the last batch if its length does not reach the batch size (default=False).
        :param sampling_strategy: the sampling strategy to apply on each dataset, can be either "random"
            or "sequential" (default="random").
        """
        self.concat_dataset = datasets if isinstance(datasets, ConcatDataset) else ConcatDataset(datasets)
        if sampling_strategy not in ["random", "sequential"]:
            raise ValueError("The sampling strategy must be one of \"random\" and \"sequential\".")
        elif sampling_strategy == "random":
            self.samplers = [iter(RandomSampler(dataset)) for dataset in self.concat_dataset.datasets]
        else:
            self.samplers = [iter(SequentialSampler(dataset)) for dataset in self.concat_dataset.datasets]

        self.bsz = bsz
        self.max_len = max([len(dataset) for dataset in self.concat_dataset.datasets])
        self.sampling_strategy = sampling_strategy
        self.drop_last = drop_last

    def __len__(self) -> int:
        n_batches = self.max_len / self.bsz * len(self.samplers)
        if self.drop_last:
            return int(n_batches)
        else:
            return int(np.ceil(n_batches))

    def __iter__(self):
        raise NotImplementedError


class HeterogeneousSampler(BatchSamplerCore):

    def __init__(self,
                 datasets: Union[ConcatDataset, List[TranslationDataset]],
                 bsz: int,
                 drop_last: bool = False,
                 sampling_strategy: str = "random",
                 weights: List[int] = None) -> None:
        """
        Builds batches in a heterogeneous way, so that there will be different translation directions in the same batch.
        :param datasets: a list of TranslationDataset or a ConcatDataset.
        :param bsz: the batch size.
        :param drop_last: whether to drop the last batch if its length does not reach the batch size (default=False).
        :param sampling_strategy: the sampling strategy to apply on each dataset, can be either "random"
            or "sequential" (default="random").
        :param weights: the sampling probability for each dataset. If None then the datasets will share the same weight
            (1 / number of datasets), otherwise the values must sum up to 1.0 (default=None).
        """
        super().__init__(datasets, bsz, drop_last, sampling_strategy)
        self.weights = weights if weights is not None else [1 / len(self.samplers)] * len(self.samplers)
        if sum(self.weights) != 1.0:
            raise ValueError("The weights must sum to 1.")

    def __iter__(self):
        batch = []
        for _ in range(self.max_len * len(self.samplers)):
            sampler_idx = np.random.choice(len(self.samplers), p=self.weights)
            try:
                sample = next(self.samplers[sampler_idx])
            except StopIteration:
                if self.sampling_strategy == "random":
                    sampler = iter(RandomSampler(self.concat_dataset.datasets[sampler_idx]))
                else:
                    sampler = iter(SequentialSampler(self.concat_dataset.datasets[sampler_idx]))

                self.samplers[sampler_idx] = sampler
                sample = next(self.samplers[sampler_idx])

            if sampler_idx == 0:
                start = 0
            else:
                start = self.concat_dataset.cumulative_sizes[sampler_idx - 1]

            batch.append(start + sample)
            if len(batch) == self.bsz:
                yield batch
                batch = []

        if not self.drop_last:
            yield batch


class HomogeneousSampler(BatchSamplerCore):

    def __init__(self,
                 datasets: Union[ConcatDataset, List[TranslationDataset]],
                 bsz: int,
                 drop_last: bool = False,
                 sampling_strategy: str = "random") -> None:
        """
        Builds batches in a homogeneous way, so that there will only a single translation direction in a batch.
        :param datasets: a list of TranslationDataset or a ConcatDataset.
        :param bsz: the batch size.
        :param drop_last: whether to drop the last batch if its length does not reach the batch size (default=False).
        :param sampling_strategy: the sampling strategy to apply on each dataset, can be either "random"
            or "sequential" (default="random").
        """
        super().__init__(datasets, bsz, drop_last, sampling_strategy)
        self._current_sampler_idx = 0

    def __iter__(self):
        batch = []
        for _ in range(self.max_len * len(self.samplers)):
            try:
                sample = next(self.samplers[self._current_sampler_idx])
            except StopIteration:
                if self.sampling_strategy == "random":
                    sampler = iter(RandomSampler(self.concat_dataset.datasets[self._current_sampler_idx]))
                else:
                    sampler = iter(SequentialSampler(self.concat_dataset.datasets[self._current_sampler_idx]))

                self.samplers[self._current_sampler_idx] = sampler
                sample = next(self.samplers[self._current_sampler_idx])

            if self._current_sampler_idx == 0:
                start = 0
            else:
                start = self.concat_dataset.cumulative_sizes[self._current_sampler_idx - 1]

            batch.append(start + sample)
            if len(batch) == self.bsz:
                self._current_sampler_idx = (self._current_sampler_idx + 1) % len(self.samplers)
                yield batch
                batch = []

        if not self.drop_last:
            self._current_sampler_idx = (self._current_sampler_idx + 1) % len(self.samplers)
            yield batch

from typing import Dict, List, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, Subset

from continualnat.data.datasets import TranslationDataset


class Buffer(Dataset):

    def __init__(self, size: int, keep_prev_examples: bool = True) -> None:
        """
        The experience replay buffer, the fixed-size version. Each experience partition has, more or less, the
        same number of examples and such examples are randomly extracted while building the training batch.
        :param size: the buffer's size.
        :param keep_prev_examples: whether to keep a subset of the previously chosen examples during the resizing of
            the partitions.
        """
        super().__init__()
        self.size = size
        self.keep_prev_examples = keep_prev_examples
        self._exps: List[List[TranslationDataset]] = []
        self._partition_size = 0

    def __len__(self) -> int:
        return self.size if len(self._exps) != 0 else 0

    def __getitem__(self, idx):
        if len(self) == 0:
            raise ValueError("You can not take any sentence pair from the buffer since it is empty.")

        # Obtain the exp from which we need to draw the example
        idx_exp = idx // self._partition_size
        exp = self._exps[idx_exp]

        # Choose randomly one of the partition's datasets
        np_generator = np.random.default_rng()
        idx_chosen_dataset = np_generator.integers(len(exp))
        chosen_dataset = exp[idx_chosen_dataset]

        # Take the example from such dataset
        idx_dataset = idx % len(chosen_dataset)
        return chosen_dataset.__getitem__(idx_dataset)

    def add_experience(self, exp: Union[TranslationDataset, List[TranslationDataset]]) -> None:
        """
        Add an experience to the buffer by adding a new partition inside of it. Keep in mind that the examples entering
        the byffer are random and that a resizing of all the previous partitions will be applied so that they will have
        approximately the same size.
        :param exp: the experience from which to draw the new examples.
        """
        # Put all the previous and current experiences together
        current_exp: List[TranslationDataset] = [exp] if isinstance(exp, TranslationDataset) else exp
        exps: List[List[TranslationDataset]] = self._exps + [current_exp]

        # Compute the new partition size and the exceeding number of examples
        self._partition_size = self.size // len(exps)
        remaining_examples_exp = self.size % len(exps)
        random_generator = np.random.default_rng()

        # Draw examples from the previous partitions and the new experience
        self._exps = []
        for exp in exps:
            # Compute the examples per dataset inside the partition and the exceeding number of them
            examples_per_dataset = self._partition_size // len(exp)
            remaining_examples_dataset = self._partition_size % len(exp)
            modified_exp = []
            for i, dataset in enumerate(exp):
                examples_to_take = examples_per_dataset

                # Distribute the exceeding examples
                if remaining_examples_exp > 0 and i == 0:
                    examples_to_take += 1
                    remaining_examples_exp -= 1

                if remaining_examples_dataset > 0 and i > 0:
                    examples_to_take += 1
                    remaining_examples_dataset -= 1

                # Choose randomly the indices to draw from the dataset
                if self.keep_prev_examples and hasattr(dataset.dataset, "indices"):
                    chosen_idxs = random_generator.choice(dataset.dataset.indices, examples_to_take, replace=False)
                else:
                    chosen_idxs = random_generator.choice(range(len(dataset)), examples_to_take, replace=False)

                # Workaround to avoid some issues
                subset = dataset.dataset if isinstance(dataset.dataset, Subset) else dataset
                dataset.dataset = Subset(subset.dataset, indices=chosen_idxs.tolist())

                # Add the dataset to the partition
                modified_exp.append(dataset)

            # Add the partition to the list of partitions
            self._exps.append(modified_exp)

    def empty(self) -> None:
        """
        Empty the buffer by removing all the partitions.
        """
        self._exps = []
        self._partition_size = 0

    def partition_sizes(self) -> Tuple[Dict[int, int], Dict[str, int]]:
        """
        Computes the partition sizes.
        :return: a tuple containing the partition sizes and the size of each partion's datasets.
        """
        partition_sizes = {}
        datasets_size = {}
        for i, exp in enumerate(self._exps):
            partition_sizes[i] = sum([len(dataset) for dataset in exp])
            for dataset in exp:
                src_lang = dataset.src_lang
                tgt_lang = dataset.tgt_lang
                datasets_size[f"{i}_{src_lang}_{tgt_lang}"] = len(dataset)

        return partition_sizes, datasets_size

    def list_datasets(self) -> List[TranslationDataset]:
        """
        List all the datasets coming from the buffer's experiences.
        :return: a list containing all the datasets used by the buffer's experiences.
        """
        buffer_datasets = []
        for exp in self._exps:
            buffer_datasets.extend(exp)

        return buffer_datasets

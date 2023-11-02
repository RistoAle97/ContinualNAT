from collections.abc import Iterable, Iterator

from torch.utils.data import BatchSampler

from continualnat.data.batch_samplers import BatchSamplerCore


class ExperienceReplaySampler(Iterable):
    def __init__(self, exp_batch_sampler: BatchSamplerCore, mem_batch_sampler: BatchSampler) -> None:
        """
        The experience replay sampler that samples
        :param exp_batch_sampler: a BatchSamplerCore sampler such that samples examples form the current experience in a
            homogeneous or heterogeneous way.
        :param mem_batch_sampler: a torch BatchSampler that samples examples from the experience buffer in a random way.
        """
        self.exp_batch_sampler = exp_batch_sampler
        self.mem_batch_sampler = mem_batch_sampler
        self._exp_batch_sampler = iter(exp_batch_sampler)
        self._mem_batch_sampler = iter(mem_batch_sampler)

    def __len__(self) -> int:
        exp_batches = len(self.exp_batch_sampler)
        mem_batches = len(self.mem_batch_sampler)
        return max([exp_batches, mem_batches])

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(len(self)):
            try:
                exp_batch = next(self._exp_batch_sampler)
            except StopIteration:
                self._exp_batch_sampler = iter(self.exp_batch_sampler)
                exp_batch = next(self._exp_batch_sampler)

            try:
                mem_batch = next(self._mem_batch_sampler)
            except StopIteration:
                self._mem_batch_sampler = iter(self.mem_batch_sampler)
                mem_batch = next(self._mem_batch_sampler)

            mem_batch = [mem_idx + len(self.exp_batch_sampler.concat_dataset) for mem_idx in mem_batch]
            yield exp_batch + mem_batch

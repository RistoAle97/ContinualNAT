import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Optimizer


class Scheduler:

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.n_steps = 0

    def _compute_lr(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def plot_lr(self, **kwargs):
        raise NotImplementedError


class TransformerScheduler:
    def __init__(self,
                 optimizer: Optimizer,
                 d_model: int,
                 warmup_steps: int) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0

    def _compute_lr(self) -> float:
        return self.d_model ** (-0.5) * min(self.n_steps ** (-0.5), self.n_steps * self.warmup_steps ** (-1.5))

    def step(self) -> None:
        self.n_steps += 1
        lr = self._compute_lr()
        for param in self.optimizer.param_groups:
            param["lr"] = lr

    @staticmethod
    def plot_lr(n_steps, d_model, warmup_steps):
        lrs = []
        for i in range(1, n_steps):
            lr = d_model ** (-0.5) * min(i ** (-0.5), i * warmup_steps ** (-1.5))
            lrs.append(lr)

        steps = np.arange(len(lrs))
        plt.plot(steps, lrs)
        plt.show()

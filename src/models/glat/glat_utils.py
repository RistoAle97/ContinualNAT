from torch import nn


class LambdaScheduler(nn.Module):

    def __init__(self, start_ratio: float = 0.5, end_ratio: float = 0.3, start: int = 0, steps: int = 300000) -> None:
        """
        Scheduler for the lambda value used in the glancing strategy. Inspired from
        https://github.com/baoy-nlp/Latent-GLAT/blob/main/latent_glat/glat.py#L316.
        :param start_ratio: the starting lambda value.
        :param end_ratio: the ending lambda value.
        :param start: at which step the scheduling should start.
        :param steps: the number of steps.
        """
        super().__init__()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.anneal_start = start
        self.anneal_end = start + steps
        self.anneal_steps = steps
        self.step_ratio = (self.end_ratio - self.start_ratio) / self.anneal_steps

    def forward(self, step):
        if step < self.anneal_start:
            return self.start_ratio
        elif step > self.anneal_end:
            return self.end_ratio
        else:
            ratio = self.start_ratio + self.step_ratio * (step - self.anneal_start)
            return ratio

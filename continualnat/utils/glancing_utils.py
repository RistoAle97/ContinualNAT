import torch


class LambdaScheduler:
    def __init__(self, start_ratio: float = 0.5, end_ratio: float = 0.2, start: int = 0, steps: int = 300000) -> None:
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
        self._anneal_start = start
        self._anneal_end = start + steps
        self.anneal_steps = steps
        self._step_ratio = (self.end_ratio - self.start_ratio) / self.anneal_steps
        self.last_ratio = start_ratio

    def __call__(self, step: int) -> float:
        if step < self._anneal_start:
            ratio = self.start_ratio
        elif step > self._anneal_end:
            ratio = self.end_ratio
        else:
            ratio = self.start_ratio + self._step_ratio * (step - self._anneal_start)

        self.last_ratio = ratio
        return ratio


class GlancingSampler:
    def __init__(self, adaptive: bool = True, strategy: str = "uniform") -> None:
        self.adaptive = adaptive
        if strategy not in ["uniform", "schedule", None]:
            raise ValueError("No correct sampling strategy was chosen, use one of \"uniform\", \"schedule\" or None.")

        self.strategy = strategy

    def __call__(
        self,
        labels: torch.Tensor,
        labels_mask: torch.Tensor,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Sampler for the glancing strategy employed by the GLAT training.
        :param labels: the tokenized ground-truth target sentences.
        :param labels_mask: the non-special tokens mask for the labels.
        :param logits: the logits coming from a softmax function on the GLAT's outputs.
        :param predictions: the predicted tokens by the model.
        :param ratio: the glancing ratio, i.e. the ratio of predicted tokens that will be substituted by
            the ground-truth tokens (default=0.5).
        :return: a map of the glanced tokens (1 glanced, 0 otherwise).
        """
        # Number of positions to be replaced
        if not self.adaptive:
            n_positions = labels.size(-1) * ratio
        else:
            distance = (predictions.ne(labels) * labels_mask).float().sum(dim=-1)
            n_positions = (distance * ratio).int()

        score = labels.clone().float().uniform_()

        # Sampling strategy
        if self.strategy == "uniform":
            score.masked_fill_(~labels_mask.bool(), 2.0)
            rank = score.sort(dim=-1)[-1]
            cutoff: torch.Tensor = torch.arange(rank.size(-1), device=rank.device).expand(*rank.size())
            cutoff = (cutoff < n_positions[:, None]).long()
            sample = cutoff.scatter(1, rank, cutoff)
        elif self.strategy == "schedule":
            prob = logits.softmax(dim=-1)
            ref_score = prob.view(-1, labels.size(-1)).contiguous().gather(0, labels.view(-1, 1)).view(*labels.size())
            sample = score.lt(ref_score) * labels_mask
        else:
            sample = torch.zeros_like(labels)

        return sample

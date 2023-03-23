import torch


class BeamSearch:

    def __init__(self,
                 batch_size: int,
                 num_beams: int,
                 device: torch.device,
                 length_penalty: float = 1.0,
                 max_length: int = None) -> None:
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.hypthoses = [BeamHypotheses(num_beams, length_penalty, max_length) for _ in range(batch_size)]
        self.is_beam_finished = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

    def is_finished(self) -> bool:
        return self.is_beam_finished.all()

    def process(self):
        pass

    def finalize(self):
        pass


class BeamHypotheses:

    def __init__(self, num_beams: int, length_penalty: float = 1.0, max_length: int = None) -> None:
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add_hypothesis(self, hypothesis: torch.Tensor, sum_log_p: float, beam_ids: torch.Tensor) -> None:
        hypothesis_score = sum_log_p / (hypothesis.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or hypothesis_score > self.worst_score:
            self.beams.append((hypothesis_score, hypothesis, beam_ids))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(hypothesis_score, self.worst_score)

    def is_finished(self, best_sum_log_p: float, current_length: int) -> bool:
        if len(self) < self.num_beams:
            return False
        else:
            return True

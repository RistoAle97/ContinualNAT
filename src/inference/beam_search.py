import torch
from collections import UserDict


class BeamSearchScorer:

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
        self.hypotheses = [BeamHypotheses(num_beams, length_penalty, max_length) for _ in range(batch_size)]
        self.is_beam_finished = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

    def is_finished(self) -> bool:
        return self.is_beam_finished.all()

    def process(self,
                input_ids: torch.Tensor,
                next_scores: torch.Tensor,
                next_tokens: torch.Tensor,
                next_idxs: torch.Tensor,
                pad_token_id: int = None,
                eos_token_id: int = None):
        cur_seq_len = input_ids.size(-1)
        next_beam_scores = torch.zeros((self.batch_size, self.num_beams), dtype=next_scores.dtype, device=self.device)
        next_beam_tokens = torch.zeros((self.batch_size, self.num_beams), dtype=next_tokens.dtype, device=self.device)
        next_beam_idxs = torch.zeros((self.batch_size, self.num_beams), dtype=next_idxs.dtype, device=self.device)

        for batch_idx, beam_hyp in enumerate(self.hypotheses):
            if self.is_beam_finished[batch_idx]:
                # Pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_idxs[batch_idx, :] = 0
                continue

            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_idx) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_idxs[batch_idx])):
                batch_beam_idx = batch_idx + next_idx
                if eos_token_id is not None and next_token.item() in eos_token_id:
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue

                    beam_hyp.add_hypothesis(input_ids[batch_beam_idx].clone(), next_score.item())
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_idxs[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.num_beams:
                    break

            cur_seq_len += 1  # add up to the length which the next_scores is calculated on
            self.is_beam_finished[batch_idx] = self.is_beam_finished[batch_idx] or beam_hyp.is_finished(
                next_scores[batch_idx].max().item())
            return UserDict({"next_beam_scores": next_beam_scores.view(-1),
                             "next_beam_tokens": next_beam_tokens.view(-1),
                             "next_beam_idxs": next_beam_idxs.view(-1)})

    def finalize(self,
                 input_ids: torch.Tensor,
                 final_scores: torch.Tensor,
                 max_length: int,
                 pad_token_id: int = None,
                 eos_token_id: int = None):
        bsz = len(self.hypotheses)
        eos_token_id = [eos_token_id]
        for batch_idx, beam_hyp in enumerate(self.hypotheses):
            if self.is_beam_finished[batch_idx]:
                continue

            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add_hypothesis(final_tokens, final_score)

            # select the best hypotheses
        sent_lengths = input_ids.new(bsz)
        best = []
        best_indices = []
        best_scores = torch.zeros(bsz, device=self.device, dtype=torch.float32)

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(bsz, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(bsz, sent_max_len)
        else:
            indices: None = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class BeamHypotheses:

    def __init__(self,
                 num_beams: int,
                 length_penalty: float = 1.0,
                 max_length: int = None,
                 early_stopping: bool = False) -> None:
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.early_stopping = early_stopping
        self.beams = []
        self.worst_score = 1e9

    def __len__(self) -> int:
        return len(self.beams)

    def add_hypothesis(self, hypothesis: torch.Tensor, sum_log_p: float) -> None:
        hypothesis_score = sum_log_p / (hypothesis.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or hypothesis_score > self.worst_score:
            self.beams.append((hypothesis_score, hypothesis))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(score, idx) for idx, (score, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(hypothesis_score, self.worst_score)

    def is_finished(self, best_sum_log_p: float) -> bool:
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_log_p / (self.max_length ** self.length_penalty)

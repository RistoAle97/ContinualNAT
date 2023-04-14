import torch
from torch.functional import F
from src.models import TransformerCore
from ..utils import generate_causal_mask, create_masks


def greedy_decoding(model: TransformerCore,
                    input_ids: torch.Tensor,
                    decoder_start_token_id: int,
                    max_new_tokens: int = 10) -> torch.Tensor:
    """
    Performs greedy search for translating tokenized input sentence, this should be used only by autoregressive
    transformers. This method is heavily inspired by the greedy decoding from Huggingface.
    :param model: the autoregressive model.
    :param input_ids: the tokenized input sentence of shape (bsz, seq_len).
    :param decoder_start_token_id: the decoder start token id, for multilingual models this should be the target
        language code id.
    :param max_new_tokens: maximum allowed new tokens (default=10).
    :return: the tokenized translated sentence.
    """
    assert max_new_tokens >= 0

    with torch.no_grad():
        # Parameters
        max_length = input_ids.shape[-1] + max_new_tokens
        device = next(model.parameters()).device
        batch_size = input_ids.shape[0]

        # Set the first token as the sos token (target language code if using mBart)
        output = torch.ones(batch_size, 1, dtype=torch.int).fill_(decoder_start_token_id).to(device)

        # Keep track of unfinished sentences (the ones for which the eos token was still not generated)
        unfinished_sentences = input_ids.new(batch_size).fill_(1)
        eos_token_id_tensor = torch.tensor([model.eos_token_id]).unsqueeze(1).to(device)

        # Encode the input tokens
        e_mask = input_ids.ne(model.pad_token_id).unsqueeze(1).to(device)
        e_output = model.encode(input_ids, e_mask)

        # Generate tokens in an autoregressive fashion
        for _ in range(max_length):
            # Obtain logits from the model's decoder
            d_pad_mask = output.ne(model.pad_token_id).unsqueeze(1).to(device)
            d_causal_mask = generate_causal_mask(output.size(-1)).to(device)
            d_mask = d_pad_mask & d_causal_mask
            logits = model.decode(output, e_output, d_mask, e_mask)

            # Compute the new tokens and concatenate them to the previously generated ones
            p_logits = F.log_softmax(logits[:, -1], dim=1)
            new_tokens = p_logits.argmax(-1)
            new_tokens = new_tokens * unfinished_sentences + 1 * (1 - unfinished_sentences)
            output = torch.cat([output, new_tokens[:, None]], dim=-1)

            # Update tensor that tracks unfinished sentences
            unfinished_sentences = unfinished_sentences.mul(new_tokens.ne(eos_token_id_tensor).prod(dim=0))

            # Terminates if an eos token has been generated for all the sentences
            if unfinished_sentences.max() == 0:
                break

        return output


def beam_decoding(model: TransformerCore,
                  input_ids: torch.Tensor,
                  decoder_start_token_id: int,
                  max_new_tokens: int = 10,
                  num_beams: int = 5,
                  beams_to_keep: int = 1) -> torch.Tensor:
    """
    Performs beams search for translating tokenized input sentence, this should be used only by autoregressive
    transformers. This method is heavily inspired by the beam_decoding form Huggingface.
    :param model: the autoregressive model.
    :param input_ids: the tokenized input sentence of shape (bsz, seq_len).
    :param decoder_start_token_id: the decoder start token id, for multilingual models this should be the target
        language code id.
    :param max_new_tokens: maximum allowed new tokens (default=10).
    :param num_beams: number of beams (default=5).
    :param beams_to_keep: number of beams that will be returned upon calling the method (default=1).
    :return: the tokenized translated sentence.
    """
    assert max_new_tokens >= 0
    assert num_beams > 0
    assert beams_to_keep > 0
    assert num_beams >= beams_to_keep

    with torch.no_grad():
        # Parameters
        max_length = input_ids.shape[-1] + max_new_tokens
        device = next(model.parameters()).device
        batch_size = input_ids.shape[0]
        vocab_size = model.vocab_size

        # The input_ids are reshaped as (bsz * beam_size, seq_len)
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)

        # Scores for each beam
        beam_scores = torch.zeros(batch_size, num_beams, dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # (bsz * beam_size)

        # Set the first token as the sos token (target language code if using mBart)
        output = torch.ones(batch_size * num_beams, 1, dtype=torch.int).fill_(decoder_start_token_id).to(device)

        # Encode the input tokens
        e_mask = input_ids.ne(model.pad_token_id).unsqueeze(1).to(device)
        e_output = model.encode(input_ids, e_mask)

        # Build the beam hypotheses and keep track of those sentences whose search has finished
        beam_hyps = [BeamHypotheses(num_beams, max_length=max_length) for _ in range(batch_size)]
        is_finished = torch.tensor([False for _ in range(batch_size)])

        # Perform beam search
        cur_seq_len = 1
        while True:
            _, d_mask = create_masks(input_ids, output, model.pad_token_id, decoder_mask="causal")
            logits = model.decode(output, e_output, d_mask, e_mask)

            # Compute the next tokens and their scores
            p_logits = F.log_softmax(logits[:, -1], dim=1)
            scores = p_logits + beam_scores[:, None].expand_as(p_logits)
            scores = scores.view(batch_size, num_beams * vocab_size)
            scores, tokens = torch.topk(scores, 2 * num_beams, dim=1)

            # Prepare the tokens and their indexes
            beam_idxs = torch.div(tokens, vocab_size, rounding_mode="floor")
            tokens = tokens % vocab_size

            # Build the tensors for hosting the scores, tokens and indexes
            next_beam_scores = torch.zeros((batch_size, num_beams), dtype=scores.dtype, device=device)
            next_beam_tokens = torch.zeros((batch_size, num_beams), dtype=tokens.dtype, device=device)
            next_beam_idxs = torch.zeros((batch_size, num_beams), dtype=beam_idxs.dtype, device=device)
            for seq_id, beam_hyp in enumerate(beam_hyps):
                if is_finished[seq_id]:
                    # Pad the batch if the eos token was already generated for the sentence
                    next_beam_scores[seq_id, :] = 0
                    next_beam_tokens[seq_id, :] = model.pad_token_id
                    next_beam_idxs[seq_id, :] = 0
                    continue

                # Next tokens for this sequence
                beam_idx = 0
                for next_token, next_score, next_id in zip(tokens[seq_id], scores[seq_id], beam_idxs[seq_id]):
                    seq_beam_id = seq_id * num_beams + next_id
                    if next_token.item() == model.eos_token_id:
                        # If the sequence has ended, then add it as a hypothesis inside the beam
                        beam_hyps[seq_id].add_hypothesis(output[seq_beam_id].clone(), next_score.item())
                    else:
                        # If the sequence has not ended, then update the tensors
                        next_beam_scores[seq_id, beam_idx] = next_score
                        next_beam_tokens[seq_id, beam_idx] = next_token
                        next_beam_idxs[seq_id, beam_idx] = seq_beam_id
                        beam_idx += 1

                    # Stop if the beam is already full
                    if beam_idx == num_beams:
                        break

                # Check if all the beam hypothesis of a sequence are finished
                beam_finished = beam_hyps[seq_id].is_finished(scores[seq_id].max().item(), cur_seq_len + 1)
                is_finished[seq_id] = is_finished[seq_id] or beam_finished

            # Update the beam scores and concatenate the new tokens to the previously generated ones
            beam_scores = next_beam_scores.view(-1)
            next_beam_tokens = next_beam_tokens.view(-1).unsqueeze(-1)
            next_beam_idxs = next_beam_idxs.view(-1)
            output = torch.cat([output[next_beam_idxs], next_beam_tokens], dim=-1)

            # Update the current sequence length and check if all the sequence beams are finished
            cur_seq_len += 1
            if is_finished.all() or cur_seq_len == max_length:
                break

        # Finalize all the open beam hypotheses and add them to the generated ones
        for seq_id, beam_hyp in enumerate(beam_hyps):
            if is_finished[seq_id]:
                continue

            for beam_id in range(num_beams):
                batch_beam_id = seq_id * num_beams + beam_id
                final_score = beam_scores[batch_beam_id].item()
                final_tokens = output[batch_beam_id]
                beam_hyp.add_hypothesis(final_tokens, final_score)

        # Choose the best beams_to_keep hypothesis for each sequence
        sentence_lengths = output.new(batch_size * beams_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * beams_to_keep, device=device, dtype=torch.float32)
        for i, beam_hyp in enumerate(beam_hyps):
            best_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(beams_to_keep):
                best_hyp_tuple = best_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sentence_lengths[beams_to_keep * i + j] = len(best_hyp)
                best.append(best_hyp)
                best_scores[beams_to_keep * i + j] = best_score

        # Batches are padded if they do not reach the maximum length
        max_sentence_length = sentence_lengths.max().item() + 1
        # max_sentence_length = min(max_sentence_length, max_length)
        decoded = output.new(batch_size * beams_to_keep, max_sentence_length)
        if sentence_lengths.min().item() != sentence_lengths.max().item():
            decoded.fill_(model.pad_token_id)

        # Put the eos token at the end of each sequence
        for i, hyp in enumerate(best):
            decoded[i, :sentence_lengths[i]] = hyp
            # if sentence_lengths[i] < max_sentence_length:
            decoded[i, sentence_lengths[i]] = model.eos_token_id

        return decoded


class BeamHypotheses:

    def __init__(self,
                 num_beams: int,
                 length_penalty: float = 1.0,
                 max_length: int = None,
                 early_stopping: bool = False) -> None:
        """
        Beam hypotheses for a single sentence, see the implementation from Huggingface for a better understanding.
        :param num_beams: the number of beams.
        :param length_penalty: the length penalty, values > 1.0 favor longer sentences, while values < 1.0 favor
            shorter ones (default=1.0).
        :param max_length: the maximum allowed length (default=None).
        :param early_stopping: stop as soon as the number of finished hypotheses is equal to the number of beams
            (default=True).
        """
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.early_stopping = early_stopping
        self.beams = []
        self.worst_score = 1e9

    def __len__(self) -> int:
        return len(self.beams)

    def add_hypothesis(self, hypothesis: torch.Tensor, sum_log_p: float) -> None:
        """
        Add a beam hypothesis to the beams' list.
        :param hypothesis: a tensor of int that represents a full generated sequence.
        :param sum_log_p: the sum of log probabilities related to the hypothesis.
        :return:
        """
        hyp_score = sum_log_p / (hypothesis.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or hyp_score > self.worst_score:
            self.beams.append((hyp_score, hypothesis))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(score, idx) for idx, (score, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(hyp_score, self.worst_score)

    def is_finished(self, best_sum_log_p: float, cur_len: int) -> bool:
        """
        Checks if it is still possible to generate hypotheses better than the current worst one. If early_stopping is
        True, then the BeamHypotheses terminates when all the num_beams beams have been generated.
        :param best_sum_log_p: current best sum of log probabilities not in the hypotheses.
        :param cur_len: the current length of the best hypothesis in terms of score.
        :return: if the no more hypotheses should be generated or not.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if self.length_penalty > 0.0:
                highest_possible_score = best_sum_log_p / (self.max_length ** self.length_penalty)
            else:
                highest_possible_score = best_sum_log_p / (cur_len ** self.length_penalty)

            return self.worst_score >= highest_possible_score

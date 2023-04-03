import torch
from torch.functional import F
from src.models import TransformerCore
from ..utils import generate_causal_mask


def greedy_decoding(model: TransformerCore,
                    input_ids: torch.Tensor,
                    decoder_start_token_id: int,
                    max_new_tokens: int = 10) -> torch.Tensor:
    """
    Performs greedy search for translating tokenized input sentence, this should be used only by autoregressive
    transformers.
    :param model: the autoregressive model.
    :param input_ids: the tokenized input sentence of shape (bsz, seq_len).
    :param decoder_start_token_id: the decoder start token id, for multilingual models this should be the target
        language code id.
    :param max_new_tokens: maximum allowed new tokens.
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
        for _ in range(1, max_length):
            # Obtain logits from the model's decoder
            d_pad_mask = output.ne(model.pad_token_id).unsqueeze(1).to(device)
            d_causal_mask = generate_causal_mask(output.size(-1)).to(device)
            d_mask = d_pad_mask & d_causal_mask
            logits = model.decode(output, e_output, d_mask, e_mask)

            # Compute the new tokens and concatenate them to the previously generated ones
            logits = F.log_softmax(logits[:, -1], dim=1)
            new_tokens = logits.argmax(-1)
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
                  sos_token_id: int,
                  eos_token_id: int,
                  pad_token_id: int,
                  max_new_tokens: int = 10,
                  beam_size: int = 5) -> torch.Tensor:
    """
    Performs beams search for translating tokenized input sentence, this should be used only by autoregressive
    transformers.
    :param model: the autoregressive model.
    :param input_ids: the tokenized input sentence of shape (bsz, seq_len).
    :param sos_token_id: start of sentence token id.
    :param eos_token_id: end of sentence token id, for multilingual models this should be the target language code id.
    :param pad_token_id: pad token id.
    :param max_new_tokens: maximum allowed new tokens.
    :param beam_size: number of beams.
    :return: the tokenized translated sentence.
    """
    assert max_new_tokens >= 0
    assert beam_size > 1

    with torch.no_grad():
        # Parameters
        max_length = input_ids.shape[-1] + max_new_tokens
        device = next(model.parameters()).device
        batch_size = input_ids.shape[0]

        # Scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        beam_scores[:, 1:] = 1e-9
        beam_scores = beam_scores.view((-1))  # (batch_size * beam_size)

        output = torch.ones(batch_size, 1, dtype=torch.int).fill_(sos_token_id).to(device)
        e_output = model.encode(input_ids)
        e_output = e_output.repeat_interleave(beam_size, dim=0)
        return output


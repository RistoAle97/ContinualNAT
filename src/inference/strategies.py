import torch
from torch.functional import F
from src.models import TransformerCore
from ..utils import generate_causal_mask


def greedy_decoding(model: TransformerCore,
                    input_ids: torch.Tensor,
                    e_pad_mask: torch.Tensor,
                    sos_token_id: int,
                    eos_token_id: int,
                    max_new_tokens: int = 10) -> torch.Tensor:
    """
    Performs greedy search for translating tokenized input sentence, this should be used only by autoregressive
    transformers.
    :param model: the autoregressive model.
    :param input_ids: the tokenized input sentence of shape (batch_size, seq_len).
    :param e_pad_mask: the padding mask for the input tensor.
    :param sos_token_id: start of sentence token id.
    :param eos_token_id: end of sentence token id, for multilingual models this should be the target language code id.
    :param max_new_tokens: maximum allowed new tokens.
    :return: the tokenized translated sentence.
    """
    with torch.no_grad():
        # Parameters
        max_length = input_ids.shape[-1] + max_new_tokens
        device = next(model.parameters()).device
        batch_size = input_ids.shape[0]

        # Set the first token as the sos token (target language code if using mBart)
        output = torch.ones(batch_size, 1, dtype=torch.int).fill_(sos_token_id).to(device)

        # Keep track of unfinished sentences (the ones for which the eos token was still not generated)
        unfinished_sentences = input_ids.new(batch_size).fill_(1)
        eos_token_id_tensor = torch.tensor([eos_token_id]).unsqueeze(1).to(device)

        # Encode the input tokens
        e_output = model.encode(input_ids)

        # Generate tokens in an autoregressive fashion
        for _ in range(1, max_length):
            # Obtain logits from the model's decoder
            tgt_mask = generate_causal_mask(output.shape[-1]).to(device)
            logits = model.decode(e_output, output, d_mask=tgt_mask, e_pad_mask=e_pad_mask)

            # Compute the new tokens and concatenate them to the previously generated ones
            logits = F.log_softmax(logits[:, -1], dim=1)
            new_tokens = logits.argmax(-1)
            new_tokens = new_tokens * unfinished_sentences + 1 * (1 - unfinished_sentences)
            output = torch.cat([output, new_tokens[:, None]], dim=-1)

            # Update tensor that tracks unfinished sentences
            unfinished_sentences = unfinished_sentences.mul(new_tokens.ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))

            # Terminates if an eos token has been generated for all the sentences
            if unfinished_sentences.max() == 0:
                break

        return output


def beam_decoding(model: TransformerCore,
                  x: torch.Tensor,
                  sos_token_id: int,
                  eos_token_id: int,
                  max_length: int = None,
                  beam_size: int = 2) -> torch.Tensor:
    with torch.no_grad():
        if max_length is None:
            max_length = x.shape[-1]

        device = next(model.parameters()).device
        x = x.to(device)
        output = torch.ones(1, 1, dtype=torch.int).fill_(sos_token_id).to(device)
        e_output = model.encode(x)
    pass

import torch
from models.TransformerCore import TransformerCore
from utilities import generate_square_subsequent_mask


def greedy_decoding(model: TransformerCore,
                    x: torch.Tensor,
                    sos_token_id: int,
                    eos_token_id: int,
                    max_length: int = None) -> torch.Tensor:
    """
    Performs greedy search for translating tokenized input sentence, this should be used only by autoregressive
    transformers.
    :param model: the autoregressive model.
    :param x: the tokenized input sentence of shape (1, seq_len).
    :param sos_token_id: start of sentence token id.
    :param eos_token_id: end of sentence token id, for multilingual models this should be the target language code id.
    :param max_length: maximum allowed length, if no value is passed then it will be the same as for the input sentence.
    :return: the tokenized translated sentence.
    """
    with torch.no_grad():
        if max_length is None:
            max_length = x.shape[-1]

        device = next(model.parameters()).device
        x = x.to(device)
        output = torch.ones(x.shape[0], 1, dtype=torch.int).fill_(sos_token_id).to(device)
        e_output = model.encode(x)
        for i in range(1, max_length):
            tgt_mask = generate_square_subsequent_mask(output.shape[-1]).to(device)
            next_p = model.decode(e_output, output, d_mask=tgt_mask)
            new_token = next_p[:, -1, :].argmax(-1)
            output = torch.cat([output, new_token.unsqueeze(-1)], dim=-1)
            if x.shape[0] == 1 and new_token == eos_token_id:
                break

        return output


def beam_decoding(model: TransformerCore,
                  x: torch.Tensor,
                  sos_token_id: int,
                  eos_token_id: int,
                  max_length: int = None,
                  beam_size: int = 2) -> torch.Tensor:
    pass

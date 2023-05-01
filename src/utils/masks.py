import torch
from typing import Tuple


def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    :param seq_len: length of the sequence to mask.
    :return: causal mask for the autoregressive decoder.
    """
    return torch.tril(torch.ones(1, seq_len, seq_len, dtype=torch.bool))


def generate_causal_nat_mask(seq_len: int) -> torch.Tensor:
    """
    Generates a diagonal matrix of -inf, in order to avoid a position from attending to itself.
    :param seq_len: length of the sequence to mask.
    :return: causal mask (with -inf on the diagonal) for the standard non-autoregressive decoder.
    """
    return torch.diag(torch.ones(1, seq_len, seq_len, dtype=torch.bool))


def create_masks(input_ids: torch.Tensor,
                 decoder_input_ids: torch.Tensor,
                 pad_token_id: int,
                 decoder_mask: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masks for both encoder and decoder. The encoder's mask will prevent the module from attending on padding
    tokens, while the decoder's mask will also prevent the module from attending on user-specificied tokens
    (e.g.: prevent the model to look ahead by using a causal mask).
    :param input_ids: the encoder's input tokens.
    :param decoder_input_ids: the decoder's input tokens.
    :param pad_token_id: the model pad token id.
    :param decoder_mask: the type of mask for the decoder, if None only the padding mask for the decoder will be built.
        The possibile values are None (only padding), causal (the model will not attend on future positions) and
        diagonal (the model will not attend on the current position). (default=None)
    :return: the masks for both the encoder and decoder of a transformer-based model.
    """
    e_mask = input_ids.ne(pad_token_id).unsqueeze(1).to(input_ids.device)
    d_pad_mask = decoder_input_ids.ne(pad_token_id).unsqueeze(1).to(decoder_input_ids.device)
    if decoder_mask not in [None, "causal", "diagonal"]:
        raise ValueError("The decoder mask should be one of None, \"causal\" and \"diagonal\"")

    seq_len = decoder_input_ids.size(-1)
    if decoder_mask == "causal":
        nopeak_mask = generate_causal_mask(seq_len).to(decoder_input_ids.device)
    elif decoder_mask == "diagonal":
        nopeak_mask = generate_causal_nat_mask(seq_len).to(decoder_input_ids.device)
    else:
        nopeak_mask = torch.ones(1, seq_len, seq_len, dtype=torch.bool, device=decoder_input_ids.device)

    d_mask = d_pad_mask & nopeak_mask  # (bsz, seq_len, seq_len)
    return e_mask, d_mask

import torch
from typing import Dict


def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    :param seq_len: length of the sequence to mask.
    :return: causal mask for the autoregressive decoder.
    """
    return torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)


def generate_causal_nat_mask(seq_len: int) -> torch.Tensor:
    """
    Generates a diagonal matrix of -inf, in order to avoid a position from attending to itself.
    :param seq_len: length of the sequence to mask.
    :return: causal mask (with -inf on the diagonal) for the standard non-autoregressive decoder.
    """
    return torch.diag(torch.ones(seq_len) * float("-inf"))


def create_masks(input_ids: torch.Tensor,
                 decoder_input_ids: torch.Tensor,
                 pad_token_id: int,
                 decoder_mask: str = None) -> Dict[str, torch.Tensor]:
    e_pad_mask = (input_ids == pad_token_id).to(input_ids.device)
    d_pad_mask = (decoder_input_ids == pad_token_id).to(decoder_input_ids.device)
    if decoder_mask not in [None, "causal", "diagonal"]:
        raise ValueError("The decoder mask should be one of None, \"causal\" and \"diagonal\"")
    match decoder_mask:
        case "causal":
            decoder_mask = generate_causal_mask(decoder_input_ids.shape[-1]).to(decoder_input_ids.device)
        case "diagonal":
            decoder_mask = generate_causal_nat_mask(decoder_input_ids.shape[-1]).to(decoder_input_ids.device)

    return {"e_pad_mask": e_pad_mask, "d_pad_mask": d_pad_mask, "d_mask": decoder_mask}

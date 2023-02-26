import torch
from torch import nn
from typing import Tuple


def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    :param seq_len: length of the sequence to mask.
    :return: causal mask for the autoregressive decoder.
    """
    return torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)


def generate_causal_nar_mask(seq_len: int) -> torch.Tensor:
    """
    Generates a diagonal matrix of -inf, in order to avoid a position from attending to itself.
    :param seq_len: length of the sequence to mask.
    :return: causal mask for the standard non-autoregressive decoder.
    """
    return torch.diag(torch.ones(seq_len) * float("-inf"))


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """
    Shift input ids one token to the right.
    :param input_ids: a tensor of shape (batch_size, seq_len) or (seq_len).
    :param pad_token_id: id of the pad token.
    :param decoder_start_token_id: start token id, which, in the case of the mBart tokenizer, is the token language.
    :return: torch.Tensor shifted to the right.
    """
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)

    shifted_input_ids = torch.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids = torch.where(shifted_input_ids == decoder_start_token_id, pad_token_id, shifted_input_ids)
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


def model_n_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Computes the number of parameters, and the trainable ones, of a pytorch model.
    :param: model: a pytorch nn.Module.
    """
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_parameters = sum(p.numel() for p in model.parameters())
    return model_parameters, trainable_parameters


def model_size(model: nn.Module, size: str = "mb") -> float:
    """
    Computes the size of a pytorch model in terms of kb, mb or gb.
    :param model: a pytorch nn.Module.
    :param size: string which defines the wanted size to compute.
    :return: the size of the model.
    """
    if size not in ["kb", "mb", "gb"]:
        raise ValueError("The size of the model can only be shown in kb, mb or gb.")

    allowed_sizes = {"kb": 1, "mb": 2, "gb": 3}
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = (param_size + buffer_size) / 1024 ** allowed_sizes[size]
    return size_all
